import torch
from qlib.utils import init_instance_by_config
from main import task, instruments_list  # 假设 instruments_list 是完整列表

def chunked_predict(model, base_task, chunk_size=100):
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    all_preds = {}
    for chunk in chunks(instruments_list, chunk_size):
        # 更新 handler 的 instruments
        base_task["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = chunk
        ds = init_instance_by_config(base_task["dataset"])
        # 批量预测
        preds = model.predict(ds, segment="test")
        for (inst, dt), v in preds.items():
            all_preds.setdefault(inst, {})[str(dt)] = float(v)
        # 释放显存
        torch.cuda.empty_cache()
    return all_preds

if __name__ == "__main__":
    # 1) 先训练好一个全量模型（或分批训练后合并权重）
    ds_full = init_instance_by_config(task["dataset"])
    # 自动填 d_feat、init model 等略...
    model = init_instance_by_config(task["model"])
    model.fit(ds_full)

    # 2) 分块预测
    results = chunked_predict(model, task, chunk_size=50)
    # 3) 保存
    import json
    with open("predictions.json", "w") as f:
        json.dump(results, f, indent=2)