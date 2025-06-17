import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random
import os
from pathlib import Path
from typing import List, Dict, Optional
import requests
from tqdm import tqdm

class QlibStockCrawler:
    """
    專門為 qlib dump_bin.py 格式設計的股票爬蟲
    """
    
    def __init__(self, output_dir: str = "./stock_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 設置請求頭避免被封
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        獲取單支股票數據，格式符合 dump_bin.py 要求
        
        Parameters:
        -----------
        symbol: str
            股票代碼 (例如: AAPL, GOOGL, 2330.TW)
        start_date: str
            開始日期 "YYYY-MM-DD"
        end_date: str  
            結束日期 "YYYY-MM-DD"
            
        Returns:
        --------
        pd.DataFrame: 包含 date, symbol, open, high, low, close, volume 等欄位
        """
        try:
            # 使用 yfinance 獲取數據
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                print(f"Warning: No data found for {symbol}")
                return None
            
            # 重置索引，將日期變為欄位
            data = data.reset_index()
            
            # 重命名欄位以符合 dump_bin.py 預期格式
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low', 
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 添加 symbol 欄位 (dump_bin.py 需要)
            data['symbol'] = symbol.upper()
            
            # 確保日期格式正確
            data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            
            # 選擇需要的欄位，按照 dump_bin.py 預期順序
            required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in data.columns]
            data = data[available_columns]
            
            # 移除任何 NaN 值
            data = data.dropna()
            
            # 確保數值欄位是 float 類型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            print(data.head())  # 顯示前幾行數據
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, 
                          symbols: List[str], 
                          start_date: str, 
                          end_date: str,
                          delay_range: tuple = (1, 3)) -> Dict[str, pd.DataFrame]:
        """
        批量獲取多支股票數據
        
        Parameters:
        -----------
        symbols: List[str]
            股票代碼列表
        start_date: str
            開始日期
        end_date: str
            結束日期
        delay_range: tuple
            請求間隔時間範圍 (避免被限制)
            
        Returns:
        --------
        Dict[str, pd.DataFrame]: 股票代碼對應的數據
        """
        all_data = {}
        
        print(f"開始獲取 {len(symbols)} 支股票數據...")
        
        for i, symbol in enumerate(tqdm(symbols, desc="獲取股票數據")):
            data = self.get_stock_data(symbol, start_date, end_date)
            
            if data is not None and not data.empty:
                all_data[symbol] = data
                print(f"✓ {symbol}: {len(data)} 筆數據")
            else:
                print(f"✗ {symbol}: 無數據")
            
            # 添加延遲避免被限制 (除了最後一個)
            if i < len(symbols) - 1:
                delay = random.uniform(*delay_range)
                time.sleep(delay)
        
        return all_data
    
    def save_for_qlib(self, all_data: Dict[str, pd.DataFrame], 
                     individual_files: bool = True,
                     combined_file: bool = True):
        """
        保存數據為 qlib dump_bin.py 可接受的格式
        
        Parameters:
        -----------
        all_data: Dict[str, pd.DataFrame]
            股票數據字典
        individual_files: bool
            是否保存個別股票文件
        combined_file: bool  
            是否保存合併文件
        """
        
        if individual_files:
            # 保存個別股票文件 (每支股票一個 CSV)
            for symbol, data in all_data.items():
                if data is not None and not data.empty:
                    # 清理文件名中的特殊字符
                    clean_symbol = symbol.replace('.', '_').replace('/', '_')
                    filename = self.output_dir / f"{clean_symbol}.csv"
                    data.to_csv(filename, index=False)
                    print(f"已保存 {symbol} 數據到 {filename}")
            
            # for fn in glob.glob(f"{self.output_dir}/{clean_symbol}.csv"):
                # 1) 读取并 parse 日期
                    df = pd.read_csv(self.output_dir / f"{clean_symbol}.csv", parse_dates=["date"])
                    # 2) 排序（可选，但有助于保持稳定）
                    df.sort_values(["symbol", "date"], inplace=True)
                    # 3) 写回（强制日期格式，也能保证后面 dump_bin 里 parse 得到 datetime）
                    df.to_csv(fn, index=False, date_format="%Y-%m-%d")
                    print(f"✔ fixed {fn}")

        if combined_file and all_data:
            # 保存合併文件 (所有股票在一個 CSV 中)
            combined_data = pd.concat(all_data.values(), ignore_index=True)
            combined_filename = self.output_dir / "all_stocks.csv"
            combined_data.to_csv(combined_filename, index=False)
            print(f"已保存合併數據到 {combined_filename}")
            
            return combined_filename
    
    def get_taiwan_stocks(self, symbols: List[str], start_date: str, end_date: str):
        """
        專門獲取台股數據
        
        Parameters:
        -----------
        symbols: List[str]
            台股代碼列表 (例如: ['2330', '2317', '2454'])
        """
        # 為台股添加 .TW 後綴
        tw_symbols = [f"{symbol}.TW" if not symbol.endswith('.TW') else symbol 
                     for symbol in symbols]
        
        return self.get_multiple_stocks(tw_symbols, start_date, end_date)
    
    def get_us_stocks(self, symbols: List[str], start_date: str, end_date: str):
        """
        專門獲取美股數據
        """
        return self.get_multiple_stocks(symbols, start_date, end_date)
    
    def validate_data_format(self, data: pd.DataFrame) -> bool:
        """
        驗證數據格式是否符合 dump_bin.py 要求
        """
        required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        # 檢查必要欄位
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            print(f"缺少必要欄位: {missing_columns}")
            return False
        
        # 檢查日期格式
        try:
            pd.to_datetime(data['date'])
        except:
            print("日期格式錯誤")
            return False
            
        # 檢查數值欄位
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                print(f"{col} 不是數值類型")
                return False
        
        print("數據格式驗證通過 ✓")
        return True

# 使用範例
if __name__ == "__main__":
    # 創建爬蟲實例
    crawler = QlibStockCrawler(output_dir="./qlib_stock_data")
    
    # 範例1: 獲取美股數據
    us_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print("=== 獲取美股數據 ===")
    us_data = crawler.get_us_stocks(us_symbols, start_date, end_date)
    
    # 保存美股數據
    us_output_file = crawler.save_for_qlib(us_data, individual_files=True, combined_file=True)
    
    # 範例2: 獲取台股數據  
    tw_symbols = ["2330", "2317", "2454", "0050", "0056"]
    
    print("\n=== 獲取台股數據 ===")
    tw_data = crawler.get_taiwan_stocks(tw_symbols, start_date, end_date)
    
    # 保存台股數據
    tw_output_file = crawler.save_for_qlib(tw_data, individual_files=True, combined_file=True)
    
    # 驗證數據格式
    if us_data:
        sample_data = list(us_data.values())[0]
        print(f"\n=== 數據格式驗證 ===")
        crawler.validate_data_format(sample_data)
        print(f"數據樣本:\n{sample_data.head()}")