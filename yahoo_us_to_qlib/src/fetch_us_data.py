import sys
from pathlib import Path
# add Qlib scripts folder to PYTHONPATH so data_collector is found
ROOT_DIR = Path(__file__).resolve().parents[2]  # workspace root (qlib_test2)
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import fire
import os
from datetime import datetime
from qlib import init
from data_collector.yahoo.collector import YahooCollectorUS1d

# Define the directory to save the collected data
SAVE_DIR = Path.home() / ".qlib" / "stock_data" / "us_data"

# Ensure the save directory exists
SAVE_DIR.mkdir(parents=True, exist_ok=True)

class StockDataCollector:
    def __init__(self, start_date: str = "2020-01-01", end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")
        self.collector = YahooCollectorUS1d(
            save_dir=SAVE_DIR,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            max_workers=4,
            max_collector_count=2,
            delay=0.5
        )

    def collect_data(self):
        print(f"Collecting US stock data from {self.start_date} to {self.end_date}...")
        self.collector.collector_data()
        print("Data collection complete.")

    def normalize_data(self):
        print("Normalizing collected data...")
        self.collector.normalize_data()
        print("Data normalization complete.")

if __name__ == "__main__":
    fire.Fire(StockDataCollector)