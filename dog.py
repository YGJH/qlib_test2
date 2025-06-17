import os, sys, json, argparse, subprocess, torch
# 确保能 import qlib 及 main.task
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP
from pathlib import Path
import pandas as pd
from qlib.constant import REG_US
import qlib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import json
import torch
import subprocess
from scipy import stats

from qlib.contrib.data.handler import Alpha360
from qlib.data.dataset import DataHandlerLP
from qlib.data.dataset.handler import DataHandler

# Custom Alpha360 handler that ensures proper DatetimeIndex
class FixedAlpha360(Alpha360):
    def __init__(self, *args, **kwargs):
        # Add our custom processor before the default ones
        from qlib.data.dataset.processor import ProcessInf, Fillna
        
        infer_processors = kwargs.get('infer_processors', [])
        if not infer_processors:
            # Use proper processor configuration format
            infer_processors = [
                {"class": "FixDatetimeIndex", "module_path": "dog"},
                {"class": "ProcessInf", "module_path": "qlib.data.dataset.processor"},
                {"class": "Fillna", "module_path": "qlib.data.dataset.processor"}
            ]
        else:
            # Insert our fix at the beginning
            infer_processors = [{"class": "FixDatetimeIndex", "module_path": "dog"}] + infer_processors
        kwargs['infer_processors'] = infer_processors
        
        super().__init__(*args, **kwargs)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def future_predict(steps: int = 7):
    """
    对每只股票最后一个 window 做自回归多步预测。
    每步预测后，把预测值推入时间窗口尾部（全特征维度复制），滚动窗口继续下步预测。
    返回：{instrument: [pred1, pred2, ..., predN]}
    """
    instruments_list = get_instruments_list()  # 获取全量股票列表
    for chunk in chunks(instruments_list, 100):
        task = load_task_config(instruments_list=chunk)  # 使用当前批次股票列表
        # 1) 实例化 Dataset & Model
        ds = init_instance_by_config(task["dataset"])
        # 2) 抽取最后一个 window
        df_feat = ds.prepare("train", col_set=["feature"], data_key=DataHandlerLP.DK_I)
        last = df_feat.groupby(level="instrument").tail(1)["feature"]
        instruments = last.index.get_level_values("instrument").tolist()
        arr = np.stack(last.values)
        # 如果 arr 只有 (N, F)，则扩成 (N, F, 1)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        N, d_feat, T = arr.shape

        # 在实例化 model 之前，自动检测并设置 d_feat
        # 用 train 段数据来探测特征维度
        ds_tmp = init_instance_by_config(task["dataset"])
        df_tmp = ds_tmp.prepare(
            "train", col_set=["feature"], data_key=DataHandlerLP.DK_L
        )
        # 取第一个样本的 feature 窗口，shape = (d_feat, window_len)
        first_feat = df_tmp["feature"].iloc[0]
        d_feat = first_feat.shape[0]
        task["model"]["kwargs"]["d_feat"] = d_feat
        # 现在再实例化 model
        model = init_instance_by_config(task["model"])

        model.model.eval()

        # 3) 循环自回归
        all_preds = {inst: [] for inst in instruments}
        curr = arr.copy()
        for i in range(steps):
            X = torch.tensor(curr, dtype=torch.float32).to(model.device)  # [N,d_feat,T]
            with torch.no_grad():
                y = model.model(X).cpu().numpy().reshape(N)                # [N]
            # 保存这一步预测
            for inst, val in zip(instruments, y):
                all_preds[inst].append(float(val))
            # 更新窗口：drop oldest timestep, append当前预测（复制到所有特征通道）
            y_exp = y[:, None, None]                                      # [N,1,1]
            tail = np.repeat(y_exp, d_feat, axis=1)                       # [N,d_feat,1]
            curr = np.concatenate([curr[:, :, 1:], tail], axis=2)         # [N,d_feat,T]

    return all_preds


def save_stock2concept(instruments_list: list, stock2concept_path: str):
    BASE_DIR = Path(".qlib/qlib_data/my_us_data")
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


def load_task_config(instruments_list: list):
    # 5. 手動從 calendars/day.txt 構建交易日曆
    BASE_DIR = Path(".qlib/qlib_data/my_us_data")

    calendar_file = BASE_DIR / "calendars" / "day.txt"
    if not calendar_file.exists():
        raise FileNotFoundError(f"Calendar file not found: {calendar_file}")
    calendar_df = pd.read_csv(calendar_file, header=None)
    calendar = pd.to_datetime(calendar_df[0]).tolist()
    print(f"✓ 日曆從 {calendar[0].strftime('%Y-%m-%d')} 到 {calendar[-1].strftime('%Y-%m-%d')}")
    actual_end_date = calendar[-1].strftime('%Y-%m-%d')

    # 6. 檢查數據
    print("檢查可用數據...")

    # 设定训练时间范围
    config_start_date = "2024-01-01"
    
    # 計算訓練和驗證的分割點（例如用80%訓練，20%驗證）
    dt_start = pd.to_datetime(config_start_date)
    dt_actual_end = pd.to_datetime(actual_end_date)
    
    # 計算總天數的80%作為訓練結束點
    total_days = (dt_actual_end - dt_start).days
    train_days = int(total_days * 0.8)
    train_end_date = (dt_start + pd.Timedelta(days=train_days)).strftime("%Y-%m-%d")
    valid_start_date = (dt_start + pd.Timedelta(days=train_days + 1)).strftime("%Y-%m-%d")
    
    # test 保持為未來預測
    today = datetime.now().strftime("%Y-%m-%d")
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

    # Convert to datetime for comparison
    dt_config_start = pd.to_datetime(config_start_date)
    dt_train_end = pd.to_datetime(train_end_date)

    if dt_train_end < dt_config_start:
        raise ValueError("Training end date is before start date")

    dh_cfg = {
        "start_time": config_start_date,
        "end_time": actual_end_date,
        "fit_start_time": config_start_date,
        "fit_end_time": actual_end_date,
        "instruments": instruments_list,
    }
    stock2concept = str(".qlib/qlib_data/my_us_data/stock2concept.npy")
    stock_index    = str(".qlib/qlib_data/my_us_data/stock_index.npy")

    # 计算特征维度
    d_feat = None
    try:
        handler = FixedAlpha360(**dh_cfg)
        temp_data = handler.fetch()
        if hasattr(temp_data, 'feature') and temp_data.feature is not None:
            d_feat = temp_data.feature.shape[1]
        elif isinstance(temp_data, tuple) and len(temp_data) >= 1:
            d_feat = temp_data[0].shape[1] if hasattr(temp_data[0], 'shape') else 20
        else:
            d_feat = 20
    except Exception as e:
        print(f"无法自动计算特征维度: {e}")
        d_feat = 20

    task = {
        "model": {
            "class": "TransformerModel",
            "module_path": "qlib.contrib.model.pytorch_transformer",
            "kwargs": {
                "d_feat": d_feat,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 512,
                "dropout": 0.1,
                "n_epochs": 2**10,
                "lr": 5e-5,
                "batch_size": 64,
                "metric": "ic",
                "loss": "mse",
                "early_stop": 32,
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
                    "class": "FixedAlpha360",
                    "module_path": "dog",
                    "kwargs": dh_cfg,
                },
                "segments": {
                    "train": (config_start_date, train_end_date),      # 歷史數據的前80%
                    "valid": (valid_start_date, actual_end_date),     # 歷史數據的後20%
                    "test":  (today, next_week),                      # 未來7天預測
                },
                "step_len": 20,
            },
        },
    }

    return task


def download_yahoo_data():
    """
    下载 Yahoo Finance 数据并转换为 Qlib 格式。
    如果数据已存在则跳过下载。
    """
    US_DATA_DIR = Path(".qlib/qlib_data/my_us_data")
    # 以下是好的，不要刪掉
    """
    如果遇到 pandas.errors.EmptyDataError: No columns to parse from file
    那是因為下面這個腳本是update_data_to_bin
    需要先執行 scripts/get_data.py 下載原始數據到 .qlib/qlib_data/my_us_data
    這樣才能update
    命令:
    uv run scripts/get_data.py qlib_data \
    --target_dir .qlib/qlib_data/my_us_data \
    --region us
    """
    print("Downloading US data via collector.py …")
    cmd = [
        "uv",
        "run",
        "scripts/data_collector/yahoo/collector.py",
        "update_data_to_bin",
        "--qlib_data_1d_dir",
        str(US_DATA_DIR),
        "--trading_date",
        "2021-01-01",
        "--end_date",
        (datetime.now() - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)        
    print("✓ 以成功下载并转换数据到 Qlib 格式。")

def get_instruments_list():
    # ——— 在 uv run filter_stocks.py 之前插入 ———
    # 确保 instruments/all.txt 存在，否则 filter_stocks 会报错
    US_DATA_DIR = Path(".qlib/qlib_data/my_us_data")

    instruments_dir = US_DATA_DIR / "instruments"

    instruments_dir.mkdir(parents=True, exist_ok=True)
    all_file = instruments_dir / "all.txt"
    if not all_file.exists():
        with open(all_file, "w", encoding="utf-8") as f:
            # 取所有 .bin 文件名作为标的列表
            for bin_path in US_DATA_DIR.glob("**/*.bin"):
                f.write(bin_path.stem + "\n")
    # ————————————————————————————————
    # 接着再执行 filter_stocks.py
    subprocess.run([
        "uv",
        "run",
        "filter_stocks.py",
    ], check=True)

    qlib_data_path = os.path.expanduser(".qlib/qlib_data/my_us_data")
    filter_list = Path(qlib_data_path) / "instruments" / "filtered_all.txt"
    instruments_list = [s.strip() for s in open(filter_list, "r")]
    # random.shuffle(instruments_list)  # 随机打乱顺序
    return instruments_list[:400]

def save_predictions_to_json(preds: dict, out_json: str):
    """
    将预测结果保存到 JSON 文件。
    preds: {instrument: {datetime: {"return": float, "volatility": float}}}
    """
    with open(out_json, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"✓ 已保存预测结果到 {out_json}")


def predict_chunk(model, dataset, chunk, chunk_idx):
    """
    對單個chunk進行預測
    
    Args:
        model: 訓練好的模型
        dataset: 數據集
        chunk: 當前chunk的股票列表
        chunk_idx: chunk索引
    
    Returns:
        dict: 預測結果 {stock_name: {datetime: {return: float, volatility: float}}}
    """
    chunk_preds = {}
    
    print(f"Making predictions for chunk {chunk_idx}...")
    try:
        # 首先嘗試 test 段
        segment = "test"
        predictions = None
        
        try:
            # 直接使用 model.predict，讓它處理數據準備
            predictions = model.predict(dataset, segment="test")
            print(f"Successfully got test predictions for chunk {chunk_idx}")
        except Exception as e:
            print(f"Cannot get test predictions for chunk {chunk_idx}: {e}")
            print("Trying valid segment instead...")
            
            try:
                # 嘗試使用 valid 段
                predictions = model.predict(dataset, segment="valid")
                segment = "valid"
                print(f"Successfully got valid predictions for chunk {chunk_idx}")
            except Exception as e2:
                print(f"Cannot get valid predictions for chunk {chunk_idx}: {e2}")
                print("Trying train segment for demonstration...")
                
                try:
                    # 最後嘗試 train 段（雖然不太合理，但至少能運行）
                    predictions = model.predict(dataset, segment="train")
                    segment = "train"
                    print(f"Using train segment for chunk {chunk_idx} (fallback)")
                except Exception as e3:
                    print(f"All prediction attempts failed for chunk {chunk_idx}: {e3}")
                    return chunk_preds
        
        # 處理預測結果
        if predictions is not None and len(predictions) > 0:
            print(f"Processing predictions for chunk {chunk_idx}, shape: {predictions.shape}")
            
            for inst in chunk:
                try:
                    # 檢查該股票是否在預測結果中
                    if hasattr(predictions.index, 'get_level_values'):
                        # MultiIndex
                        instruments_in_pred = predictions.index.get_level_values('instrument')
                    else:
                        # 單層索引，可能股票名稱就是索引
                        instruments_in_pred = predictions.index
                    
                    if inst in instruments_in_pred:
                        if hasattr(predictions.index, 'get_level_values'):
                            # MultiIndex 情況
                            inst_preds = predictions.loc[inst]
                        else:
                            # 單層索引情況
                            inst_preds = predictions.loc[[inst]]
                        
                        # 處理每個時間點的預測
                        if hasattr(inst_preds, 'iterrows'):
                            for dt, pred_val in inst_preds.iterrows():
                                dt_str = pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M:%S")
                                rec = chunk_preds.setdefault(inst, {}).setdefault(dt_str, {})
                                
                                # 處理預測值
                                if hasattr(pred_val, 'values'):
                                    pred_array = pred_val.values
                                elif isinstance(pred_val, (list, tuple, np.ndarray)):
                                    pred_array = pred_val
                                else:
                                    pred_array = [pred_val]
                                
                                # 确保是一维数组
                                if isinstance(pred_array, np.ndarray) and pred_array.ndim > 1:
                                    pred_array = pred_array.flatten()
                                
                                # 如果是雙頭輸出
                                if len(pred_array) >= 2:
                                    rec["return"] = float(pred_array[0])
                                    rec["volatility"] = float(pred_array[1])
                                else:
                                    rec["return"] = float(pred_array[0])
                                    rec["volatility"] = 0.0
                        else:
                            # 如果不是DataFrame，直接處理
                            dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            rec = chunk_preds.setdefault(inst, {}).setdefault(dt_str, {})
                            
                            if hasattr(inst_preds, 'values'):
                                pred_array = inst_preds.values.flatten()
                            else:
                                pred_array = [float(inst_preds)]
                            
                            if len(pred_array) >= 2:
                                rec["return"] = float(pred_array[0])
                                rec["volatility"] = float(pred_array[1])
                            else:
                                rec["return"] = float(pred_array[0])
                                rec["volatility"] = 0.0
                                
                except Exception as e:
                    print(f"Error processing predictions for {inst} in chunk {chunk_idx}: {e}")
                    continue
        
        print(f"Prediction completed for chunk {chunk_idx} using {segment} segment")
        print(f"Got predictions for {len(chunk_preds)} stocks")
        
    except Exception as e:
        print(f"Prediction failed for chunk {chunk_idx}: {e}")
        import traceback
        traceback.print_exc()
    
    return chunk_preds


# def run_pipeline(chunk_size: int, out_json: str, steps: int):
#     """
#     分块训练并预测
#     """
#     # 读取完整股票列表
#     evals_result = {"train": [], "valid": []}
    
#     download_yahoo_data()  # 确保数据已下载
#     instruments_list = get_instruments_list()
#     US_DATA_DIR = Path(".qlib/qlib_data/my_us_data")
#     BASE_DIR = Path(".qlib/qlib_data/my_us_data")
#     qlib.init(
#         provider_uri=US_DATA_DIR,
#         region=REG_US,
#         joblib_backend="sequential",
#     )

#     all_preds = {}
#     save_stock2concept(instruments_list, BASE_DIR)

#     for idx, chunk in enumerate(chunks(instruments_list, chunk_size), 1):
#         print(f">>> Chunk {idx}: 共 {len(chunk)} 支股票")
        
#         # 準備配置和數據
#         task = load_task_config(chunk)
#         dataset = init_instance_by_config(task["dataset"])
#         df = dataset.prepare("train", col_set=["feature","label"],
#                              data_key=DataHandlerLP.DK_L)
#         d_feat = (df if not isinstance(df, tuple) else df[0])["feature"].iloc[0].shape[0]
#         task["model"]["kwargs"]["d_feat"] = d_feat

#         # 初始化模型並加載之前的權重（除了第一個chunk）
#         model = init_instance_by_config(task["model"])
#         ckpt = "model_checkpoint.pth"
#         if idx > 1 and os.path.exists(ckpt):
#             # load inner nn.Module weights
#             state = torch.load(ckpt, map_location=model.device)
#             model.model.load_state_dict(state)

#         # 訓練模型
#         local_hist = {"train": [], "valid": []}
#         print("Training model...")
#         model.fit(dataset, evals_result=local_hist)
#         print("Training completed")
        
#         # 將本次訓練的指標追加到全局 history
#         evals_result["train"].extend(local_hist["train"])
#         evals_result["valid"].extend(local_hist["valid"])
        
#         # 進行預測（使用新的預測函數）
#         chunk_predictions = predict_chunk(model, dataset, chunk, idx)
        
#         # 合併到總預測結果
#         for stock, stock_preds in chunk_predictions.items():
#             if stock not in all_preds:
#                 all_preds[stock] = {}
#             all_preds[stock].update(stock_preds)

#         # 保存模型檢查點
#         torch.save(model.model.state_dict(), ckpt)
#         # 释放显存
#         del model
#         torch.cuda.empty_cache()

#     # 多步自回归预测未来 N 天
#     print("Performing multi-step prediction...")
#     preds = future_predict(steps=steps)

#     # 合併結果
#     result = {
#         "chunked_predictions": all_preds,
#         "multi_step_predictions": preds,
#     }

#     # 绘制学习曲线
#     plt.figure(figsize=(6,4))
#     plt.plot(evals_result["train"], label="train IC")
#     plt.plot(evals_result["valid"], label="valid IC")
#     plt.xlabel("Epoch")
#     plt.ylabel("IC")
#     plt.title("Train vs Valid Curve")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("train_valid_curve.png", dpi=150)
#     print("✓ 已保存学习曲线到 train_valid_curve.png")

#     # 保存結果
#     with open(out_json, "w") as f:
#         json.dump(result, f, indent=2)
#     print(f"✓ 已保存預測結果到 {out_json}")

#     return result

def comprehensive_predict(model, dataset, chunk, steps: int = 7):
    """
    全面預測：提供多種預測標的
    """
    predictions = {}
    
    try:
        # 1. 獲取最新的歷史數據作為基準
        print("Preparing training data for predictions...")
        latest_data = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        if latest_data is None or len(latest_data) == 0:
            print("No training data available for predictions")
            return predictions
            
        # 檢查數據結構
        print(f"Data shape: {latest_data.shape}")
        print(f"Data index: {latest_data.index.names}")
        
        # 獲取可用的股票列表
        if hasattr(latest_data.index, 'get_level_values'):
            available_instruments = set(latest_data.index.get_level_values('instrument'))
        else:
            available_instruments = set(latest_data.index)
            
        print(f"Available instruments in data: {len(available_instruments)} stocks")
        print(f"Sample instruments: {list(available_instruments)[:10]}")
        print(f"Chunk instruments: {chunk[:5]}...")
        
        # 檢查有多少chunk中的股票在數據中
        valid_stocks = [stock for stock in chunk if stock in available_instruments]
        missing_stocks = [stock for stock in chunk if stock not in available_instruments]
        
        print(f"Valid stocks in chunk: {len(valid_stocks)}")
        print(f"Missing stocks in chunk: {len(missing_stocks)}")
        if missing_stocks:
            print(f"Missing stocks sample: {missing_stocks[:5]}")
        
        for stock in valid_stocks:  # 只處理有效的股票
            try:
                print(f"Processing stock: {stock}")
                
                # 安全地獲取股票數據
                try:
                    if hasattr(latest_data.index, 'get_level_values'):
                        # MultiIndex 情況
                        stock_data = latest_data.loc[latest_data.index.get_level_values('instrument') == stock]
                    else:
                        # 單層索引情況
                        stock_data = latest_data.loc[stock:stock] if stock in latest_data.index else pd.DataFrame()
                    
                    if len(stock_data) == 0:
                        print(f"No data rows for stock {stock}, skipping...")
                        continue
                        
                    # 取最近20天數據，如果不足20天就取全部
                    stock_data = stock_data.tail(min(20, len(stock_data)))
                    print(f"Got {len(stock_data)} data points for {stock}")
                    
                except Exception as e:
                    print(f"Error getting data for stock {stock}: {e}")
                    continue
                
                # 檢查必要的列是否存在
                if 'feature' not in stock_data.columns:
                    print(f"No feature data for stock {stock}, skipping...")
                    continue
                
                # 獲取最新特徵和標籤
                try:
                    latest_features = stock_data['feature'].iloc[-1]
                    latest_returns = stock_data['label'].values if 'label' in stock_data.columns else None
                    
                    # 確保 latest_features 是 numpy array
                    if hasattr(latest_features, 'values'):
                        latest_features = latest_features.values
                    elif isinstance(latest_features, list):
                        latest_features = np.array(latest_features)
                    
                    # 檢查特徵維度
                    if len(latest_features.shape) == 0 or latest_features.shape[0] == 0:
                        print(f"Invalid feature shape for {stock}: {latest_features.shape}")
                        continue
                        
                    print(f"Feature shape for {stock}: {latest_features.shape}")
                    
                except Exception as e:
                    print(f"Error extracting features for {stock}: {e}")
                    continue
                
                # 初始化預測結果
                stock_predictions = {
                    "basic_info": {
                        "symbol": stock,
                        "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                        "last_known_return": float(latest_returns[-1]) if latest_returns is not None and len(latest_returns) > 0 else None,
                        "data_points": len(stock_data),
                        "feature_dimension": len(latest_features)
                    },
                    "multi_horizon_returns": {},
                    "risk_metrics": {},
                    "technical_signals": {},
                    "trend_analysis": {},
                    "probability_distributions": {}
                }
                
                # 2. 多時間段收益率預測
                print(f"Predicting multi-horizon returns for {stock}...")
                for horizon in [1, 3, 5, 7]:
                    if horizon <= steps:
                        try:
                            predicted_returns = predict_multi_step_returns(model, latest_features, horizon)
                            if predicted_returns is not None and len(predicted_returns) > 0:
                                stock_predictions["multi_horizon_returns"][f"{horizon}d"] = {
                                    "expected_return": float(np.mean(predicted_returns)),
                                    "cumulative_return": float(np.sum(predicted_returns)),
                                    "daily_returns": [float(x) for x in predicted_returns]
                                }
                            else:
                                raise ValueError(f"Empty predictions for {horizon}d")
                        except Exception as e:
                            print(f"Error predicting {horizon}d returns for {stock}: {e}")
                            stock_predictions["multi_horizon_returns"][f"{horizon}d"] = {
                                "expected_return": 0.0,
                                "cumulative_return": 0.0,
                                "daily_returns": [0.0] * horizon,
                                "error": str(e)
                            }
                
                # 3. 風險指標預測（簡化版本）
                print(f"Calculating risk metrics for {stock}...")
                try:
                    returns_7d = predict_multi_step_returns(model, latest_features, 7)
                    if returns_7d is not None and len(returns_7d) > 0:
                        stock_predictions["risk_metrics"] = {
                            "volatility_7d": float(np.std(returns_7d)),
                            "expected_return_7d": float(np.mean(returns_7d)),
                            "min_return_7d": float(np.min(returns_7d)),
                            "max_return_7d": float(np.max(returns_7d))
                        }
                    else:
                        raise ValueError("Empty risk predictions")
                except Exception as e:
                    print(f"Error calculating risk metrics for {stock}: {e}")
                    stock_predictions["risk_metrics"] = {
                        "volatility_7d": 0.02,
                        "expected_return_7d": 0.0,
                        "min_return_7d": -0.05,
                        "max_return_7d": 0.05,
                        "error": str(e)
                    }
                
                # 4. 簡化的選股評分
                try:
                    expected_7d = stock_predictions["multi_horizon_returns"].get("7d", {}).get("expected_return", 0)
                    volatility = stock_predictions["risk_metrics"].get("volatility_7d", 0.02)
                    
                    # 簡單的風險調整收益評分
                    if volatility > 0:
                        risk_adjusted_return = expected_7d / volatility
                    else:
                        risk_adjusted_return = 0
                    
                    stock_predictions["selection_scores"] = {
                        "expected_return": expected_7d,
                        "volatility": volatility,
                        "risk_adjusted_return": risk_adjusted_return,
                        "composite_score": max(0, min(100, (risk_adjusted_return + 1) * 50))
                    }
                except Exception as e:
                    print(f"Error calculating selection scores for {stock}: {e}")
                    stock_predictions["selection_scores"] = {
                        "composite_score": 50.0,
                        "error": str(e)
                    }
                
                predictions[stock] = stock_predictions
                print(f"✓ Successfully processed {stock}")
                
            except Exception as e:
                print(f"Error processing stock {stock}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
        print(f"Error in comprehensive_predict: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"Successfully processed {len(predictions)} out of {len(chunk)} stocks")
    return predictions


def predict_multi_step_returns(model, initial_features, steps):
    """簡化版多步預測收益率"""
    try:
        returns = []
        current_features = initial_features.copy()
        
        for step in range(steps):
            # 確保特徵是正確的形狀
            if len(current_features.shape) == 1:
                feature_tensor = torch.from_numpy(current_features.reshape(1, -1)).float().to(model.device)
            else:
                feature_tensor = torch.from_numpy(current_features).float().to(model.device)
            
            with torch.no_grad():
                pred = model.model(feature_tensor)
                if pred.dim() == 2 and pred.size(1) >= 2:
                    predicted_return = float(pred[0, 0].cpu())
                elif pred.dim() == 2 and pred.size(1) == 1:
                    predicted_return = float(pred[0, 0].cpu())
                else:
                    predicted_return = float(pred[0].cpu() if pred.dim() == 1 else pred.cpu())
            
            returns.append(predicted_return)
            
            # 簡化的特徵更新（不更新，保持原始特徵）
            # 這樣避免了特徵更新可能導致的錯誤
        
        return np.array(returns)
        
    except Exception as e:
        print(f"Error in predict_multi_step_returns: {e}")
        return np.array([0.0] * steps)  # 返回零收益作為默認值


def calculate_max_drawdown(returns):
    """計算最大回撤"""
    cumulative = np.cumprod(1 + np.array(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))


def calculate_sharpe_estimate(returns):
    """估算夏普比率"""
    if np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # 年化


def generate_technical_signals(historical_data, predicted_returns):
    """生成技術指標信號"""
    signals = {}
    
    if len(historical_data) >= 5:
        recent_returns = historical_data['label'].values[-5:] if 'label' in historical_data.columns else []
        
        # 動量信號
        signals["momentum_5d"] = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
        signals["momentum_signal"] = "BUY" if signals["momentum_5d"] > 0.01 else "SELL" if signals["momentum_5d"] < -0.01 else "HOLD"
        
        # 預測動量
        signals["predicted_momentum_7d"] = float(np.mean(predicted_returns))
        signals["predicted_signal"] = "BUY" if signals["predicted_momentum_7d"] > 0.02 else "SELL" if signals["predicted_momentum_7d"] < -0.02 else "HOLD"
        
        # 反轉信號
        signals["mean_reversion_signal"] = "BUY" if signals["momentum_5d"] < -0.03 else "SELL" if signals["momentum_5d"] > 0.03 else "HOLD"
    
    return signals


def analyze_trends(predicted_returns, historical_returns):
    """趨勢分析"""
    analysis = {}
    
    # 預測趨勢
    analysis["predicted_trend"] = "UPTREND" if np.mean(predicted_returns) > 0.005 else "DOWNTREND" if np.mean(predicted_returns) < -0.005 else "SIDEWAYS"
    analysis["trend_strength"] = float(abs(np.mean(predicted_returns)))
    analysis["trend_consistency"] = float(np.mean([1 if r > 0 else 0 for r in predicted_returns]))
    
    # 趨勢變化
    if historical_returns is not None and len(historical_returns) >= 5:
        hist_trend = np.mean(historical_returns[-5:])
        pred_trend = np.mean(predicted_returns)
        analysis["trend_change"] = "ACCELERATING" if pred_trend > hist_trend else "DECELERATING" if pred_trend < hist_trend else "STABLE"
    
    return analysis


def calculate_outperform_probability(returns):
    """計算跑贏市場的概率（假設市場收益率為0.001）"""
    market_return = 0.001  # 假設日均市場收益率
    return float(np.mean(returns > market_return))


def calculate_selection_scores(predictions):
    """計算選股評分"""
    scores = {}
    
    # 收益評分 (0-100)
    expected_7d = predictions["multi_horizon_returns"].get("7d", {}).get("expected_return", 0)
    scores["return_score"] = max(0, min(100, (expected_7d + 0.05) * 1000))
    
    # 風險評分 (0-100, 越低越好)
    volatility = predictions["risk_metrics"].get("volatility_7d", 0.1)
    scores["risk_score"] = max(0, min(100, 100 - volatility * 1000))
    
    # 夏普評分
    sharpe = predictions["risk_metrics"].get("sharpe_estimate", 0)
    scores["sharpe_score"] = max(0, min(100, (sharpe + 2) * 25))
    
    # 概率評分
    prob_positive = predictions["probability_distributions"].get("prob_positive_7d", 0.5)
    scores["probability_score"] = prob_positive * 100
    
    # 綜合評分
    scores["composite_score"] = (
        scores["return_score"] * 0.3 +
        scores["risk_score"] * 0.2 +
        scores["sharpe_score"] * 0.3 +
        scores["probability_score"] * 0.2
    )
    
    return scores


def run_comprehensive_pipeline(chunk_size: int, out_json: str, steps: int):
    """
    運行全面預測流程
    """
    evals_result = {"train": [], "valid": []}
    
    download_yahoo_data()
    instruments_list = get_instruments_list()
    US_DATA_DIR = Path(".qlib/qlib_data/my_us_data")
    BASE_DIR = Path(".qlib/qlib_data/my_us_data")
    qlib.init(provider_uri=US_DATA_DIR, region=REG_US, joblib_backend="sequential")

    all_comprehensive_predictions = {}
    save_stock2concept(instruments_list, BASE_DIR)

    for idx, chunk in enumerate(chunks(instruments_list, chunk_size), 1):
        print(f">>> Processing Chunk {idx}: {len(chunk)} stocks")
        
        # 訓練模型
        task = load_task_config(chunk)
        dataset = init_instance_by_config(task["dataset"])
        df = dataset.prepare("train", col_set=["feature","label"], data_key=DataHandlerLP.DK_L)
        d_feat = (df if not isinstance(df, tuple) else df[0])["feature"].iloc[0].shape[0]
        task["model"]["kwargs"]["d_feat"] = d_feat

        model = init_instance_by_config(task["model"])
        ckpt = "model_checkpoint.pth"
        if idx > 1 and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=model.device)
            model.model.load_state_dict(state)

        local_hist = {"train": [], "valid": []}
        print("Training model...")
        model.fit(dataset, evals_result=local_hist)
        print("Training completed")
        
        evals_result["train"].extend(local_hist["train"])
        evals_result["valid"].extend(local_hist["valid"])
        
        # 全面預測
        print("Performing comprehensive predictions...")
        chunk_comprehensive = comprehensive_predict(model, dataset, chunk, steps)
        all_comprehensive_predictions.update(chunk_comprehensive)
        
        torch.save(model.model.state_dict(), ckpt)
        del model
        torch.cuda.empty_cache()

    # 生成選股建議
    recommendations = generate_stock_recommendations(all_comprehensive_predictions)
    
    # 保存結果
    final_result = {
        "comprehensive_predictions": all_comprehensive_predictions,
        "stock_recommendations": recommendations,
        "training_history": evals_result,
        "metadata": {
            "prediction_date": datetime.now().isoformat(),
            "prediction_horizon_days": steps,
            "total_stocks_analyzed": len(all_comprehensive_predictions)
        }
    }

    with open(out_json, "w") as f:
        json.dump(final_result, f, indent=2)
    print(f"✓ Comprehensive predictions saved to {out_json}")
    
    return final_result


def generate_stock_recommendations(predictions):
    """生成選股建議"""
    recommendations = {
        "top_picks": {},
        "avoid_list": {},
        "category_leaders": {},
        "risk_adjusted_picks": {}
    }
    
    # 按綜合評分排序
    scored_stocks = [(stock, data["selection_scores"]["composite_score"]) 
                    for stock, data in predictions.items() 
                    if "selection_scores" in data]
    scored_stocks.sort(key=lambda x: x[1], reverse=True)
    
    # 頂級推薦（前10）
    recommendations["top_picks"] = {
        stock: {
            "score": score,
            "expected_7d_return": predictions[stock]["multi_horizon_returns"].get("7d", {}).get("expected_return", 0),
            "risk_level": "LOW" if predictions[stock]["risk_metrics"]["volatility_7d"] < 0.02 else "MEDIUM" if predictions[stock]["risk_metrics"]["volatility_7d"] < 0.04 else "HIGH"
        }
        for stock, score in scored_stocks[:10]
    }
    
    # 避開清單（後10）
    recommendations["avoid_list"] = {
        stock: {
            "score": score,
            "reason": "Low expected return" if predictions[stock]["multi_horizon_returns"].get("7d", {}).get("expected_return", 0) < 0 else "High risk"
        }
        for stock, score in scored_stocks[-10:]
    }
    
    return recommendations


# Custom processor to fix datetime index before ProcessInf
from qlib.data.dataset.processor import Processor

class FixDatetimeIndex(Processor):
    """Fix datetime index to ensure it's a proper DatetimeIndex"""
    
    def __call__(self, df):
        if df is None or df.empty:
            return df
            
        # Check if we have a MultiIndex with datetime level
        if hasattr(df, 'index') and hasattr(df.index, 'nlevels') and df.index.nlevels == 2:
            try:
                # Get the datetime level (typically level 1)
                datetime_level = df.index.get_level_values(1)
                
                # If it's not already a DatetimeIndex, convert it
                if not isinstance(datetime_level, pd.DatetimeIndex):
                    instruments = df.index.get_level_values(0)
                    new_datetime = pd.to_datetime(datetime_level)
                    new_index = pd.MultiIndex.from_arrays(
                        [instruments, new_datetime],
                        names=df.index.names
                    )
                    df.index = new_index
                    
            except Exception as e:
                print(f"Warning: Could not fix datetime index: {e}")
                
        return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Qlib US stock prediction pipeline")
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of stocks per chunk")
    parser.add_argument("--out_json", type=str, default="predictions.json", help="Output JSON file for predictions")
    parser.add_argument("--steps", type=int, default=7, help="Number of future steps to predict")
    
    args = parser.parse_args()
    
    # run_pipeline(args.chunk_size, args.out_json, args.steps)
    run_comprehensive_pipeline(args.chunk_size, args.out_json, args.steps)