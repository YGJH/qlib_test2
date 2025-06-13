import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The 'axis' keyword in DataFrame.groupby is deprecated and will be removed *")
import traceback

import sys
import subprocess
from pathlib import Path
import json
import numpy as np
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
from qlib.data.dataset.handler import DataHandlerLP
# 1. 下载官方 Qlib 美股 `.bin` 数据（如本地不存在才下载）
US_DATA_DIR = Path.home() / ".qlib" / "qlib_data" / "my_us_data"
# # 以下是好的，不要刪掉
# if not US_DATA_DIR.exists() or not any(US_DATA_DIR.glob("*.bin")):
#     print("Downloading US data via get_yahoo_data.py …")
#     subprocess.run([
#         "uv",
#         "run",
#         "get_yahoo_data.py",
#     ], check=True)
#     print("✓ Yahoo data fetched and dumped to Qlib format")
# else:
#     print("Qlib-format US data already present, skipping download.")

import random
subprocess.run([
    "uv",
    "run",
    "filter_stocks.py",
], check=True)
qlib_data_path = os.path.expanduser("/home/charles/.qlib/qlib_data/my_us_data")
filter_list = Path(qlib_data_path) / "instruments" / "filtered_all.txt"
instruments_list = [s.strip() for s in open(filter_list, "r")]
# instruments_list = instruments_list[:min(len(instruments_list) , 50)] # 限制最多 50 支股票
instruments_list = random.sample(instruments_list,500) # 限制最多 50 支股票

# 2-3. 直接初始化 Qlib 指向官方 `us_data` 目录
qlib.init(
    provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "my_us_data"),
    region=REG_US,
    instruments_list = instruments_list,
    joblib_backend="sequential",
)
# 4. 手动递归查找 us_data 下所有 .bin 文件作为标的
BASE_DIR = Path.home() / ".qlib" / "qlib_data" / "my_us_data"
bin_files = list(BASE_DIR.glob("**/*.bin"))
print(f"✓ 讀取到 {len(instruments_list)} 支股票")
print(f"  股票代號: {instruments_list[:10]}{'...' if len(instruments_list) > 10 else ''}")

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
config_start_date = "2024-01-01"
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

start_ts = pd.Timestamp(config_start_date)
end_ts = pd.Timestamp(final_actual_end_str)

dh_cfg = {
    "start_time": config_start_date ,
    "end_time": final_actual_end_str,
    "fit_start_time": config_start_date,
    "fit_end_time": final_actual_end_str,
    "instruments": instruments_list,
}
stock2concept = str(Path.home()/".qlib/qlib_data/my_us_data/stock2concept.npy")
stock_index    = str(Path.home()/".qlib/qlib_data/my_us_data/stock_index.npy")

task = {
    "model": {
        "class": "HIST",
        "module_path": "qlib.contrib.model.pytorch_hist",
        "kwargs": {
            "d_feat": 512,           # Alpha360 的输入维度
            "hidden_size": 128,
            "num_layers": 30,
            "dropout": 0.1,
            "n_epochs": 2**16,
            "lr": 0.0005,
            "metric": "ic",
            "early_stop": 15,
            "loss": "mse",
            "base_model": "GRU",
            "stock2concept": stock2concept,
            "stock_index": stock_index,
            "optimizer": "adam",
            "GPU": 0,
            "seed": 42,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": dh_cfg,
            },
            "segments": {
                "train": (config_start_date, final_actual_end_str),
            },
            "step_len": 20,
        },
    },
}
instruments_list = sorted(instruments_list)  # 保持顺序稳定
stock_index_dict = {sym: idx for idx, sym in enumerate(instruments_list)}
np.save(BASE_DIR / "stock_index.npy", stock_index_dict)
print(f"Saved stock_index.npy with {len(stock_index_dict)} symbols.")
sector_map = {}  # sector_name -> list of instrument indices
for idx, sym in enumerate(instruments_list):
    try:
        info = yf.Ticker(sym).info
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    sector_map.setdefault(sector, []).append(idx)

concepts = sorted(sector_map.keys())
stock2concept = np.zeros((len(instruments_list), len(concepts)), dtype=int)
for j, sector in enumerate(concepts):
    stock2concept[sector_map[sector], j] = 1

# 保存并同时把概念名称写入一个对照文件（可选）
np.save(BASE_DIR / "stock2concept.npy", stock2concept)
with open(BASE_DIR / "concept_names.txt", "w") as f:
    for sec in concepts:
        f.write(sec + "\n")

print(f"Saved stock2concept.npy with shape {stock2concept.shape}.")
print(f"Concept names written to concept_names.txt.")


import torch

# 在模型 fit 之前打印一下

print(f"\n開始訓練模型...")
# print(f"  DH Cfg: {dh_cfg}")
print(f"  訓練段: {task['dataset']['kwargs']['segments']['train']}")

# 簡化模型配置 - 使用全部資料訓練

# export CUDA_VISIBLE_DEVICES=0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 确保只使用第一个 GPU
# 强制覆盖 segments，确保绝对没有第三个字符串
task["dataset"]["kwargs"]["segments"] = {
    "train": (config_start_date, final_actual_end_str),
    "valid": (config_start_date, final_actual_end_str),
    "test":  (config_start_date, final_actual_end_str),
}


import time
import random


timestamp = int(time.time())
random_id = random.randint(1000, 9999)
experiment_name = f"hist_train_us_new_{timestamp}_{random_id}"
recorder_name = f"hist_rec_new_{timestamp}_{random_id}"

print(f"使用全新实验名称: {experiment_name}")
WORK_DIR = Path(f"./snapshot_hist_clean_{random_id}/")

try:
    import mlflow
    # --- 1) 清理残留的 snapshot（checkpoint）目录 和 MLflow Run ---
    if mlflow.active_run() is not None:
        mlflow.end_run()

    import logging
    # --- 2) 开啓更详细的日志 ---
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("qlib").setLevel(logging.DEBUG)

    # --- 3) 彻底清理所有可能的 checkpoint ---
    from pathlib import Path
    import shutil

    # 清理可能的 checkpoint 位置
    checkpoint_locations = [
        Path(WORK_DIR) / "hist_us.pth",
        Path("./hist_us.pth"),
        Path("./snapshot_hist_clean_v8/hist_us.pth"), 
        Path.cwd() / "hist_us.pth",
        Path.home() / ".qlib" / "hist_us.pth",
    ]

    for ckpt_path in checkpoint_locations:
        if ckpt_path.exists():
            ckpt_path.unlink()
            print(f"✓ 删除旧 checkpoint: {ckpt_path}")

    # 完全删除并重建工作目录
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
        print(f"✓ 删除整个工作目录: {WORK_DIR}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ 重新创建工作目录: {WORK_DIR}")


    # --- 4) 准备 Dataset (先于 Model) ---
    from qlib.utils import init_instance_by_config
    dataset = init_instance_by_config(task["dataset"])

    # 拿一下一个样本的 feature 维度
    sample = dataset.prepare(
        "train", col_set=["feature"], data_key=DataHandlerLP.DK_I
    )
    # 如果 prepare 返回的是 (df_train, df_valid) 之类的 tuple，取第一个
    if isinstance(sample, tuple):
        sample_df = sample[0]
    else:
        sample_df = sample
    
    # sample_df["feature"] 是个 ndarray，形状 [d_feat, T]
    d_feat = sample_df["feature"].iloc[0].shape[0]
    print(f"✓ 自动检测到特征维度 d_feat = {d_feat}")

    # 写回 model.kwargs
    task["model"]["kwargs"]["d_feat"] = d_feat

    # --- 再实例化 Model ---
    model = init_instance_by_config(task["model"])

    # --- 5) 真正跑训练 ---
    print(">> 开始训练")
    
    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
        # 1) 记录超参时剔除 instruments 列表，避免过长截断
        params = flatten_dict(task)
        params.pop("dataset.kwargs.handler.kwargs.instruments", None)
        R.log_params(**params)

        # 2) 把 instruments_list 写入文件，并用 mlflow.log_artifact 记录
        # instr_file = "instruments.txt"
        # with open(instr_file, "w", encoding="utf-8") as f:
            # for sym in instruments_list:
                # f.write(sym + "\n")
        # import mlflow
        # mlflow.log_artifact(instr_file)
        print(">> 開始fit模型")
        # 3) 真正跑训练
        model.fit(dataset)
        R.save_objects(hist_model=model)
        print(">> 训练结束")
except Exception as e:
    print(f"訓練過程出錯: {e}")
    import traceback
    traceback.print_exc()



# 替换原来的预测代码部分

# 7. 用训练好的模型预测未来一周并保存为 JSON
from datetime import timedelta
import json

print("\n开始预测未来7天...")

try:
    # 构造未来一周的日期范围
    last_date = pd.to_datetime(actual_end_date)
    future_start = last_date + timedelta(days=1)
    future_end = last_date + timedelta(days=9)

    future_start_str = future_start.strftime("%Y-%m-%d")
    future_end_str = future_end.strftime("%Y-%m-%d")
    
    print(f"预测日期范围: {future_start_str} 到 {future_end_str}")
    
    # 使用最近的数据作为特征进行预测
    recent_end = last_date
    recent_start = recent_end - timedelta(days=30)
    
    recent_start_str = recent_start.strftime("%Y-%m-%d")
    recent_end_str = recent_end.strftime("%Y-%m-%d")
    
    print(f"使用最近数据: {recent_start_str} 到 {recent_end_str}")
    
    # 构建最近数据的配置
    recent_dh_cfg = {
        "start_time": recent_start_str,
        "end_time": recent_end_str,
        "fit_start_time": config_start_date,
        "fit_end_time": final_actual_end_str,
        "instruments": instruments_list,
    }
    
    recent_segment = {
        "test": (recent_start_str, recent_end_str)
    }
    
    recent_dataset_cfg = {
        "class": "DatasetH", 
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": recent_dh_cfg
            },
            "segments": recent_segment,
            "step_len": 20,
        },
    }
    
    # 创建最近数据集
    recent_dataset = init_instance_by_config(recent_dataset_cfg)
    
    # 使用训练好的模型进行预测
    predictions = model.predict(recent_dataset)
    
    # 准备增强版预测结果数据
    prediction_results = {
        "prediction_date_range": {
            "start": future_start_str,
            "end": future_end_str,
            "note": "基于最近历史数据的外推预测"
        },
        "model_info": {
            "model_type": "HIST",
            "train_period": f"{config_start_date} to {final_actual_end_str}",
            "num_stocks": len(instruments_list),
            "prediction_based_on": f"{recent_start_str} to {recent_end_str}"
        },
        "market_summary": {},
        "daily_predictions": {},
        "stock_analysis": {}
    }
    
    # 整理预测结果
    if predictions is not None and not predictions.empty:
        print(f"✓ 获得预测结果，数据形状: {predictions.shape}")
        
        unique_dates = predictions.index.get_level_values('datetime').unique()
        
        # 初始化 trend_data 和 latest_predictions，避免后续分支未定义
        trend_data = {}
        latest_predictions = {}
        
        # 获取最近几天的预测结果进行趋势分析
        if len(unique_dates) >= 30:
            # 获取最近30天的预测值用于趋势计算
            recent_30_dates = unique_dates[-30:]
            
            for instrument in instruments_list:
                daily_preds = []
                for date in recent_30_dates:
                    try:
                        pred_value = predictions.loc[(date, instrument)]
                        if pd.notna(pred_value):
                            daily_preds.append(float(pred_value))
                    except (KeyError, IndexError):
                        continue
                
                if len(daily_preds) >= 2:
                    trend_data[instrument] = daily_preds
                    latest_predictions[instrument] = daily_preds[-1]
        elif len(unique_dates) > 0:
            # 如果数据不足30天，使用最新一天初始化 trend_data
            latest_date = unique_dates[-1]
            for instrument in instruments_list:
                try:
                    pred_value = predictions.loc[(latest_date, instrument)]
                    if pd.notna(pred_value):
                        latest_predictions[instrument] = float(pred_value)
                        trend_data[instrument] = [float(pred_value)]
                except (KeyError, IndexError):
                    continue
        
        # 分析函数
        def analyze_trend(values):
            """分析价格趋势"""
            if len(values) < 2:
                return "NEUTRAL", 0.0
            
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            avg_change = sum(changes) / len(changes)
            
            if avg_change > 0.005:  # 0.5%以上
                return "BULLISH", abs(avg_change)
            elif avg_change < -0.005:  # -0.5%以下
                return "BEARISH", abs(avg_change)
            else:
                return "NEUTRAL", abs(avg_change)
        
        def generate_signal(pred_value, trend, confidence):
            """生成交易信号"""
            if trend == "BULLISH" and confidence > 0.01:
                return "BUY" if pred_value > 0 else "HOLD"
            elif trend == "BEARISH" and confidence > 0.01:
                return "SELL" if pred_value < 0 else "HOLD"
            else:
                return "HOLD"
        
        def get_risk_level(confidence, volatility):
            """评估风险等级"""
            risk_score = confidence * 0.7 + volatility * 0.3
            if risk_score > 0.02:
                return "HIGH"
            elif risk_score > 0.01:
                return "MEDIUM"
            else:
                return "LOW"
        
        # 为未来7天生成详细预测
        import numpy as np
        np.random.seed(42)
        
        all_predictions = []
        
        for i in range(7):
            future_date = future_start + timedelta(days=i)
            future_date_str = future_date.strftime('%Y-%m-%d')
            
            daily_data = {
                "date": future_date_str,
                "stocks": {},
                "market_stats": {}
            }
            
            daily_predictions = []
            daily_signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
            for instrument, base_pred in latest_predictions.items():
                # 获取趋势信息
                if instrument in trend_data and len(trend_data[instrument]) > 1:
                    trend, trend_strength = analyze_trend(trend_data[instrument])
                    volatility = np.std(trend_data[instrument]) if len(trend_data[instrument]) > 1 else 0.01
                else:
                    trend, trend_strength = "NEUTRAL", 0.01
                    volatility = 0.01
                
                # 添加时间衰减和随机扰动
                time_decay = 0.99 ** i  # 随时间衰减
                trend_factor = 1 + (trend_strength * (1 if trend == "BULLISH" else -1 if trend == "BEARISH" else 0))
                noise = np.random.normal(0, volatility * 0.5)  # 基于历史波动率的噪声
                
                future_pred = base_pred * time_decay * trend_factor + noise
                
                # 生成信号
                signal = generate_signal(future_pred, trend, trend_strength)
                daily_signals[signal] += 1
                
                # 计算置信度
                confidence = min(trend_strength * 100, 95)  # 转换为百分比，最高95%
                
                # 风险评估
                risk_level = get_risk_level(trend_strength, volatility)
                
                # 价格目标（简单计算）
                if signal == "BUY":
                    price_target = f"+{(future_pred * 100):.2f}%"
                elif signal == "SELL":
                    price_target = f"{(future_pred * 100):.2f}%"
                else:
                    price_target = f"{(future_pred * 100):+.2f}%"
                
                stock_data = {
                    "prediction": round(future_pred, 6),
                    "signal": signal,
                    "trend": trend,
                    "confidence": round(confidence, 2),
                    "risk_level": risk_level,
                    "price_target": price_target,
                    "volatility": round(volatility * 100, 2)  # 转换为百分比
                }
                
                daily_data["stocks"][instrument] = stock_data
                daily_predictions.append(future_pred)
            
            # 计算市场整体统计
            if daily_predictions:
                daily_data["market_stats"] = {
                    "avg_prediction": round(np.mean(daily_predictions), 6),
                    "market_sentiment": max(daily_signals, key=daily_signals.get),
                    "bullish_ratio": round(daily_signals["BUY"] / len(daily_predictions) * 100, 1),
                    "bearish_ratio": round(daily_signals["SELL"] / len(daily_predictions) * 100, 1),
                    "neutral_ratio": round(daily_signals["HOLD"] / len(daily_predictions) * 100, 1),
                    "total_stocks": len(daily_predictions)
                }
            
            prediction_results["daily_predictions"][future_date_str] = daily_data
            all_predictions.extend(daily_predictions)
        
        # 生成整体市场摘要
        if all_predictions:
            prediction_results["market_summary"] = {
                "overall_sentiment": "BULLISH" if np.mean(all_predictions) > 0.005 else "BEARISH" if np.mean(all_predictions) < -0.005 else "NEUTRAL",
                "avg_7day_prediction": round(np.mean(all_predictions), 6),
                "prediction_range": {
                    "min": round(np.min(all_predictions), 6),
                    "max": round(np.max(all_predictions), 6)
                },
                "volatility": round(np.std(all_predictions) * 100, 2),
                "recommendation": "谨慎乐观" if np.mean(all_predictions) > 0 else "保守观望"
            }
        
        # 生成个股分析摘要
        for instrument in instruments_list:  # 只分析前20支股票
            if instrument in latest_predictions:
                pred_values = []
                signals = []
                
                for day_data in prediction_results["daily_predictions"].values():
                    if instrument in day_data["stocks"]:
                        pred_values.append(day_data["stocks"][instrument]["prediction"])
                        signals.append(day_data["stocks"][instrument]["signal"])
                
                if pred_values:
                    most_common_signal = max(set(signals), key=signals.count)
                    trend_direction = "上涨" if np.mean(pred_values) > 0.005 else "下跌" if np.mean(pred_values) < -0.005 else "横盘"
                    
                    prediction_results["stock_analysis"][instrument] = {
                        "7day_avg_prediction": round(np.mean(pred_values), 6),
                        "dominant_signal": most_common_signal,
                        "trend_direction": trend_direction,
                        "prediction_consistency": round(len([s for s in signals if s == most_common_signal]) / len(signals) * 100, 1),
                        "volatility": round(np.std(pred_values) * 100, 2)
                    }
        
        print(f"✓ 预测完成，覆盖 {len(prediction_results['daily_predictions'])} 天")
        if prediction_results["daily_predictions"]:
            total_stocks = sum(len(day_data["stocks"]) for day_data in prediction_results["daily_predictions"].values())
            avg_stocks = total_stocks / len(prediction_results["daily_predictions"])
            print(f"  平均每天预测 {avg_stocks:.1f} 支股票")
            print(f"  整体市场情绪: {prediction_results['market_summary']['overall_sentiment']}")
            
    else:
        print("✗ 预测结果为空")
    
    # 保存预测结果到 JSON 文件
    output_file = f"enhanced_predictions_{timestamp}_{random_id}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(prediction_results, f, indent=2, ensure_ascii=False)
    output_file2 = f"../qlib_web/src/future.json"
    with open(output_file2, "w" , encoding='utf-8') as f:
        json.dump(prediction_results, f , indent=2, ensure_ascii=False)
    output_file2 = f"../qlib_web/public/future.json"
    with open(output_file2, "w" , encoding='utf-8') as f:
        json.dump(prediction_results, f , indent=2, ensure_ascii=False)
    
    print(f"✓ 增强预测结果已保存到: {output_file}")
    
    # 显示详细预测结果摘要
    if prediction_results.get("market_summary"):
        print(f"\n📊 市场摘要:")
        print(f"  整体情绪: {prediction_results['market_summary']['overall_sentiment']}")
        print(f"  7日平均预测: {prediction_results['market_summary']['avg_7day_prediction']:.6f}")
        print(f"  市场波动率: {prediction_results['market_summary']['volatility']:.2f}%")
        print(f"  投资建议: {prediction_results['market_summary']['recommendation']}")
    
    if prediction_results.get("daily_predictions"):
        print(f"\n📅 每日预测样例 (前3天):")
        for i, (date, day_data) in enumerate(prediction_results["daily_predictions"].items()):
            if i >= 3:
                break
            print(f"  {date}:")
            print(f"    市场情绪: {day_data['market_stats']['market_sentiment']}")
            print(f"    看涨比例: {day_data['market_stats']['bullish_ratio']}%")
            print(f"    看跌比例: {day_data['market_stats']['bearish_ratio']}%")
            
            # 显示前3支股票的详细预测
            for j, (stock, stock_data) in enumerate(day_data["stocks"].items()):
                if j >= 3:
                    break
                print(f"    {stock}: {stock_data['signal']} | 目标: {stock_data['price_target']} | 置信度: {stock_data['confidence']}%")
                
except Exception as e:
    print(f"预测过程出错: {e}")
    import traceback
    traceback.print_exc()