import sys
from pathlib import Path
# add Qlib scripts folder to sys.path so data_collector is found
ROOT_DIR = Path(__file__).resolve().parent.parent  # qlib_test2 root
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import fire
from datetime import datetime
from qlib import init
from data_collector.yahoo.collector import YahooCollectorUS1d, YahooNormalize1d
import pandas as pd

class StockDataCollector:
    def __init__(
        self,
        source_dir: str = str(Path.home() / ".qlib" / "stock_data" / "source"),
        normalize_dir: str = str(Path.home() / ".qlib" / "stock_data" / "normalize"),
        qlib_data_dir: str = str(Path.home() / ".qlib" / "qlib_data" / "us_data"),
        start_date: str = "2020-01-01",
        end_date: str = None,
    ):
        self.source_dir = Path(source_dir).expanduser()
        self.normalize_dir = Path(normalize_dir).expanduser()
        self.qlib_data_dir = Path(qlib_data_dir).expanduser()
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime("%Y-%m-%d")

        # Initialize Qlib
        init(provider_uri=str(self.qlib_data_dir))

    def download_data(self):
        collector = YahooCollectorUS1d(
            save_dir=self.source_dir,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            max_workers=4,
            delay=0.5
        )
        collector.collector_data()
        print(f"Data downloaded to {self.source_dir}")

    def normalize_data(self):
        normalizer = YahooNormalize1d()
        for csv_file in self.source_dir.glob("*.csv"):
            df = normalizer.normalize(pd.read_csv(csv_file))
            normalized_file_path = self.normalize_dir / csv_file.name
            df.to_csv(normalized_file_path, index=False)
            print(f"Normalized data saved to {normalized_file_path}")

    def run(self):
        self.download_data()
        self.normalize_data()

if __name__ == "__main__":
    fire.Fire(StockDataCollector)