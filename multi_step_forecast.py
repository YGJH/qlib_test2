import os, sys, json, argparse
import numpy as np
import torch
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from main import task

# 把项目根加入 PATH，确保能 import qlib & main
ROOT = os.path.abspath(os.path.join(__file__, ".."))
sys.path.insert(0, ROOT)

def load_all_instruments():
    fn = os.path.join(ROOT, ".qlib/qlib_data/my_us_data", "instruments", "all.txt")
    with open(fn) as f:
        return [s.strip() for s in f if s.strip()]

def forecast_multi_step(
    instruments: list = None, steps: int = 7, batch_size: int = 128
):
    if instruments is None:
        instruments = load_all_instruments()
    # 每批 instruments 数量，分批能降低内存占用
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # 实例化 Dataset & Model
    model = init_instance_by_config(task["model"])
    model.model.eval()

    all_preds = {}
    for chunk in chunks(instruments, batch_size):
        # 动态更新 instruments
        task["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = chunk
        ds = init_instance_by_config(task["dataset"])
        # 拿最后一条 window
        df_feat = ds.prepare("train", col_set=["feature"], data_key=DataHandlerLP.DK_I)
        last = df_feat.groupby(level="instrument").tail(1)["feature"]
        insts = last.index.get_level_values("instrument").tolist()
        arr = np.stack(last.values)
        # 有时每个 feature 是 1D（no time‐axis），补一个维度
        if arr.ndim == 2:
            arr = arr[:, :, None]
        N, d_feat, T = arr.shape

        curr = arr.copy()
        for i in range(steps):
            X = torch.tensor(curr, dtype=torch.float32).to(model.device)
            with torch.no_grad():
                y = model.model(X).cpu().numpy().reshape(-1)
            # 记录第 i 步预测
            for inst, val in zip(insts, y):
                all_preds.setdefault(inst, []).append(float(val))
            # 滚动窗口：丢旧、加新
            tail = np.repeat(y[:, None, None], curr.shape[1], axis=1)
            curr = np.concatenate([curr[:, :, 1:], tail], axis=2)

    return all_preds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--out", type=str, default="future_preds.json")
    args = parser.parse_args()

    preds = forecast_multi_step(
        instruments=None, steps=args.steps, batch_size=args.batch_size
    )
    with open(args.out, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Saved {len(preds)} instruments × {args.steps} steps to {args.out}")