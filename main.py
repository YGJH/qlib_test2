import sys
import subprocess
from pathlib import Path
import json
import datetime  # already imported? ensure this is present
import os            

import qlib
from qlib.constant import REG_US
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

# 2. 若還沒下載資料，就用 get_data.py 一次抓完
provider_uri = "~/.qlib/qlib_data/us_data"
if not exists_qlib_data(provider_uri):
    subprocess.run(
        [
            "uv run",
            "scripts/get_data.py",
            "qlib_data_us",
            "--target_dir",
            os.path.expanduser(provider_uri),
            "--region",
            "us",
            "--interval",
            "1d",
            "--version",
            "v2",
        ],
        check=True,
    )

# 3. 初始化 Qlib
qlib.init(provider_uri=provider_uri, region=REG_US)

# 4. 定義 market、時間範圍（動態到今天）
market   = "csi300"
benchmark= "SH000300"
today = datetime.date.today().strftime("%Y-%m-%d")
dh_cfg = {
    "start_time":     "2008-01-01",
    "end_time":       today,
    "fit_start_time": "2008-01-01",
    "fit_end_time":   today,
    "instruments":    market,
}

# 5. 配置訓練任務：模型 + 資料集
task = {
    "model": {
        "class":       "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            # ...（可自行調整超參數）...
        },
    },
    "dataset": {
        "class":       "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class":       "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": dh_cfg,
            },
            # 用最新資料做訓練 & 回測
            "segments": {
                "train": ("2008-01-01", today),
                "test":  ("2008-01-01", today),
            },
        },
    },
}

# 6. 實例化模型與資料集
model   = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# 7. 訓練階段
with R.start(experiment_name="train_exp"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    # 把訓練好的 model 存到 recorder
    R.save_objects(trained_model=model)
    train_recorder_id = R.get_recorder().id

# 8. 預測 & 回測
with R.start(experiment_name="backtest_exp"):
    # 載入剛剛訓練的模型
    train_rec = R.get_recorder(recorder_id=train_recorder_id, experiment_name="train_exp")
    model = train_rec.load_object("trained_model")
    # 8.1 產生交易訊號
    sr = SignalRecord(model=model, dataset=dataset, recorder=R.get_recorder())
    sr.generate()
    # 8.2 產生投組績效
    par = PortAnaRecord(
        recorder=R.get_recorder(),
        # (可自行加回測參數配置)
    )
    par.generate()
    backtest_recorder_id = R.get_recorder().id

# 9. 匯出所有資料為 JSON
#    從最後的 recorder 抓取 signal / performance / ts / positions
rec = R.get_recorder(recorder_id=backtest_recorder_id, experiment_name="backtest_exp")
signal_df = rec.load_object("signal.pkl")
perf_df   = rec.load_object("performance.pkl")
ts_df     = rec.load_object("ts_backtest.pkl")
pos_df    = rec.load_object("position.pkl")

export_data = {
    "signals":    signal_df.reset_index().to_dict(orient="records"),
    "metrics":    perf_df.reset_index().to_dict(orient="records"),
    "timeseries": ts_df.reset_index().to_dict(orient="records"),
    "positions":  pos_df.reset_index().to_dict(orient="records"),
}

# write into the same folder as main.py
output_path = Path(__file__).parent / "qlib_analysis_results.json"
with output_path.open("w", encoding="utf-8") as fp:
    json.dump(export_data, fp, ensure_ascii=False, indent=2)

print(f"所有結果已匯出到 {output_path.resolve()}")
