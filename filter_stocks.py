import os
import pandas as pd
from qlib.data import D
from pathlib import Path
from qlib.constant import REG_US
import qlib

# Initialize Qlib
qlib_data_path = os.path.expanduser(".qlib/qlib_data/my_us_data")
qlib.init(
    provider_uri=str(".qlib/qlib_data/my_us_data"),
    region=REG_US,
    joblib_backend="sequential",
)

# Paths
instruments_file = Path(qlib_data_path) / "instruments" / "all.txt"

filtered_instruments_file = Path(qlib_data_path) / "instruments" / "filtered_all.txt"

# Date range
start_date = "2010-01-01"
end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

# Read instruments
if not instruments_file.exists():
    print(f"Error: {instruments_file} does not exist.")
    exit(1)

with instruments_file.open("r", encoding="utf-8") as f:
    stocks = [line.strip().split()[0].lower() for line in f if line.strip()]

# Filter stocks
valid_stocks = []
for stock in stocks:
    try:
        data = D.features([stock], ["$close"], start_time=start_date, end_time=end_date, freq="day")
        if data.empty:
            print(f"No data for {stock} in the range {start_date} to {end_date}.")
        else:
            # print(f"Data for {stock}: {data.head()}")
            valid_stocks.append(stock)
    except Exception as e:
        print(f"Error loading data for {stock}: {e}")

# Write filtered instruments
with open(filtered_instruments_file, "w", encoding="utf-8") as f:
    for stock in valid_stocks:
        f.write(f"{stock}\n")

print(f"Filtered instruments written to {filtered_instruments_file}")
