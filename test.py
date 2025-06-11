import pandas as pd
import pickle
import qlib
from qlib.data import D
calendar = pd.read_pickle("/home/charles/.qlib/qlib_data/us_stocks_yahoo/calendars/day.pkl")
print(calendar.min(), calendar.max())

symbol = "aapl"  # 替換為你要檢查的股票代碼
file_path = f"/home/charles/.qlib/qlib_data/us_stocks_yahoo/features/{symbol}/close.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)
print(data.head())
calendar = pd.read_pickle("/home/charles/.qlib/qlib_data/us_stocks_yahoo/calendars/day.pkl")
print(calendar.min(), calendar.max())
# Date range
start_date = "2010-01-01"
end_date = "2025-06-11"
qlib.init(provider_uri="/home/charles/.qlib/qlib_data/us_stocks_yahoo", region="us", joblib="sequential")
print(f"Checking data for {symbol}...")

data = D.features(["aapl"], ["close"], start_time="2010-01-01", end_time="2025-06-11", freq="day")
if data.empty:
    print("No data available for AAPL.")
else:
    print(data.head())