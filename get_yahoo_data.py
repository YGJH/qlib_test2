#!/usr/bin/env python3
# prepare_qlib_data.py
import os
import sys
import subprocess
from datetime import datetime

import pandas as pd
import yfinance as yf

# 用户可根据需要修改
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
RAW_DIR = os.path.expanduser("~/qlib_csv/raw")
PROC_DIR = os.path.expanduser("~/qlib_csv/processed")
QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/my_us_data")
INCLUDE_FIELDS = "open,close,high,low,volume,factor"


def get_sp500_tickers() -> list[str]:
    """从 Wikipedia 抓取 S&P 500 成分股列表"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    # 有些代码里含有点（如 BRK.B），yfinance 需要替换为 BRK-B
    symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    print(f"Fetched {len(symbols)} tickers from S&P 500.")
    return symbols


def download_price_data(symbols: list[str], start: str, end: str, out_dir: str):
    """批量下载 Yahoo Finance 数据"""
    os.makedirs(out_dir, exist_ok=True)
    for sym in symbols:
        fn = os.path.join(out_dir, f"{sym}.csv")
        # 已下载且非空则跳过
        if os.path.exists(fn) and os.path.getsize(fn) > 1000:
            print(f"  ↳ {sym} 已存在，跳过")
            continue
        print(f"Downloading {sym} ...")
        df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            print(f"  !! Warning: {sym} returned no data, skip.")
            continue
        df.reset_index(inplace=True)
        df.to_csv(fn, index=False)


def process_csv(raw_dir: str, out_dir: str):
    """把 Yahoo CSV 重命名并补全 factor，输出到新的文件夹"""
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(raw_dir):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(raw_dir, fn)
        df = pd.read_csv(path, parse_dates=["Date"], dayfirst=False)
        # 如果“Adj Close”不存在，退回到“Close”
        if "Adj Close" in df.columns:
            close_col = "Adj Close"
        else:
            close_col = "Close"
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            close_col: "close",
            "Volume": "volume"
        })
        df["symbol"] = fn[:-4]
        df["factor"] = 1.0
        df = df[["date", "symbol", "open", "high", "low", "close", "volume", "factor"]]
        out_path = os.path.join(out_dir, fn)
        df.to_csv(out_path, index=False)
        print(f"  ↳ processed {fn}")


def dump_to_qlib(csv_path: str, qlib_path: str, fields: str):
    """调用 Qlib 的 dump_bin.py 生成 .bin 数据"""
    cmd = [
        "uv",
        "run",
        "scripts/dump_bin.py",
        "dump_all",
        "--csv_path", csv_path,
        "--qlib_dir", qlib_path,
        "--include_fields", fields
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    print("1) 抓取成分股列表")
    symbols = get_sp500_tickers()

    print("\n2) 下载历史价格到", RAW_DIR)
    download_price_data(symbols, START_DATE, END_DATE, RAW_DIR)

    print("\n3) 清洗并重命名 CSV 到", PROC_DIR)
    process_csv(RAW_DIR, PROC_DIR)

    print("\n4) 转换为 Qlib 二进制格式到", QLIB_DIR)
    dump_to_qlib(PROC_DIR, QLIB_DIR, INCLUDE_FIELDS)

    print("\n✅ All done! 在 Qlib 中初始化时使用：")
    print(f"   qlib.init(provider_uri=\"{QLIB_DIR}\", region=REG_US)  # or REG_CN")


if __name__ == "__main__":
    main()
