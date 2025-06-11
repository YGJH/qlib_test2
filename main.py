import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import subprocess
from pathlib import Path
import json
import datetime
import os
import pandas as pd
import shutil

import qlib
from qlib.constant import REG_US
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data import D

# 1. 下载官方 Qlib 美股 `.bin` 数据（如本地不存在才下载）
US_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "my_us_data"
# if not US_DATA_DIR.exists() or not any(US_DATA_DIR.glob("*.bin")):
#     print("Downloading official Qlib US data...")
#     subprocess.run([
#         sys.executable,
#         str(Path(__file__).parent / "scripts" / "get_data.py"),
#         "qlib_data",
#         "--target_dir",
#         str(US_DATA_DIR),
#         "--region",
#         "us",
#     ], check=True)
#     print("✓ Official US data download complete")
# else:
#     print("Official Qlib US data already present, skipping download.")
if not US_DATA_DIR.exists() or not any(US_DATA_DIR.glob("*.bin")):
    print("Downloading US data via get_yahoo_data.py …")
    subprocess.run([
        "uv",
        "run",
        "get_yahoo_data.py",
    ], check=True)
    print("✓ Yahoo data fetched and dumped to Qlib format")
else:
    print("Qlib-format US data already present, skipping download.")





# 2-3. 直接初始化 Qlib 指向官方 `us_data` 目录
qlib.init(
    provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "my_us_data"),
    region=REG_US,
    joblib_backend="sequential",
)
# qlib.init(provider_uri="/home/charles/.qlib/qlib_data/my_us_data", region=REG_US)
# 4. 手动递归查找 us_data 下所有 .bin 文件作为标的
BASE_DIR = Path.home() / ".qlib" / "qlib_data" / "my_us_data"
bin_files = list(BASE_DIR.glob("**/*.bin"))
instruments_list = [p.stem.lower() for p in bin_files]
print(f"✓ 讀取到 {len(instruments_list)} 支股票")
# print(f"  股票代號: {instruments_list[:10]}{'...' if len(instruments_list) > 10 else ''}")

# 5. 手动从 calendars/day.txt 构建交易日历
calendar_file = BASE_DIR / "calendars" / "day.txt"
if not calendar_file.exists():
    print(f"✗ 找不到日曆檔案: {calendar_file}")
    sys.exit(1)
calendar_df = pd.read_csv(calendar_file, header=None)
calendar = pd.to_datetime(calendar_df[0]).tolist()
print(f"✓ 日曆從 {calendar[0].strftime('%Y-%m-%d')} 到 {calendar[-1].strftime('%Y-%m-%d')}")
actual_end_date = calendar[-1].strftime('%Y-%m-%d')

# 6. 檢查數據
print("檢查可用數據...")
actual_end_date = calendar[-1].strftime('%Y-%m-%d')

# 设定训练时间范围：训练结束设置为实际最新日期
config_start_date = "2010-01-01"
train_end_str = actual_end_date  # use latest available date
# no separate test period (use all data for training)


# Convert to datetime for comparison
dt_config_start = pd.to_datetime(config_start_date)
dt_train_end = pd.to_datetime(train_end_str)
dt_actual_end = pd.to_datetime(actual_end_date)

# Adjust segments to be within the actual data range [dt_config_start, dt_actual_end]
dt_train_end = min(dt_train_end, dt_actual_end)

if dt_train_end < dt_config_start:
    print(f"警告: 訓練結束 ({dt_train_end.date()}) 在配置開始 ({dt_config_start.date()}) 之前。")
    dt_train_end = dt_config_start

final_train_end_str = dt_train_end.strftime("%Y-%m-%d")
final_actual_end_str = dt_actual_end.strftime("%Y-%m-%d")


dh_cfg = {
    "start_time": config_start_date,
    "end_time": final_actual_end_str,
    "fit_start_time": config_start_date,
    "fit_end_time": final_actual_end_str,
    "instruments": instruments_list,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "early_stopping_rounds": 50,
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": dh_cfg,
            },            "segments": {
                "train": (config_start_date, final_actual_end_str),
            },
            "step_len": 20,
        },
    },
}

print(f"\n開始訓練模型...")
# print(f"  DH Cfg: {dh_cfg}")
print(f"  訓練段: {task['dataset']['kwargs']['segments']['train']}")

# 簡化模型配置 - 使用全部資料訓練


try:
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    with R.start(experiment_name="us_stock_train"):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        train_recorder_id = R.get_recorder().id

    print("✓ 模型訓練完成")
    
    # 簡單輸出模型訓練結果
    print("✓ 訓練完成，模型已保存到 mlruns 目錄")
    print(f"✓ 實驗 ID: {train_recorder_id}")

except Exception as e:
    print(f"訓練過程出錯: {e}")
    import traceback
    traceback.print_exc()

print("程序執行完成！")

# 7. 用训练好的模型预测未来一周并保存为 JSON
from datetime import timedelta
import json

# 构造未来一周的日期范围
last_date = pd.to_datetime(actual_end_date)
future_start = last_date + timedelta(days=1)
future_end = last_date + timedelta(days=7)
future_segment = {"future": (future_start.strftime("%Y-%m-%d"), future_end.strftime("%Y-%m-%d"))}

# 构建未来数据集配置
future_dataset_cfg = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": dh_cfg},
        "segments": future_segment,
        "step_len": 20,
    },
}
future_dataset = init_instance_by_config(future_dataset_cfg)
pred_future = model.predict(future_dataset)

# 转成 JSON 并写文件
pred_df = pred_future.reset_index()
out_path = Path("future_predictions.json")
pred_df.to_json(out_path, orient="records", date_format="iso")
print(f"✓ 未来一周预测已保存到 {out_path}")

