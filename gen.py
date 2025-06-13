import numpy as np
from pathlib import Path
import yfinance as yf
from qlib.data import D
import os
import qlib
from qlib.constant import REG_US

qlib.init(
    provider_uri=str(Path.home() / ".qlib" / "qlib_data" / "my_us_data"),
    region=REG_US,
    joblib_backend="sequential",
)
qlib_data_path = os.path.expanduser("/home/charles/.qlib/qlib_data/my_us_data")
filter_list = Path(qlib_data_path) / "instruments" / "filtered_all.txt"
instruments_list = [s.strip() for s in open(filter_list, "r")]

# 如果你想再按交易日期过滤一次，也可以这么做：
# instruments_list = D.list_instruments(
#     instruments=instruments_list,
#     start_time="2000-01-01",
#     end_time="2025-06-10",
#     as_list=True
# )

# 2. 限制最多 10 只
instruments_list = instruments_list[:10]

# 3. 保存成 stock_index.npy
base = Path.home() / ".qlib/qlib_data/my_us_data"
base.mkdir(parents=True, exist_ok=True)
stock_index = np.array(instruments_list, dtype="<U10")
np.save(base / "stock_index.npy", stock_index)
print(f"Saved {len(stock_index)} symbols to stock_index.npy")
instruments = sorted(instruments_list)  # 保持顺序稳定
stock_index = np.array(instruments, dtype='<U10')  # 根据你的代码长度调整 dtype
np.save(base / "stock_index.npy", stock_index)
print(f"Saved stock_index.npy with {len(stock_index)} symbols.")

# 3. 生成 stock2concept.npy
#    这里示例用 Yahoo Finance 的 sector 字段做概念，
#    也可以替换成 industry、GICS 列表甚至自定义分组
sector_map = {}  # sector_name -> list of instrument indices
for idx, sym in enumerate(instruments):
    try:
        info = yf.Ticker(sym).info
        sector = info.get("sector") or "Unknown"
    except Exception:
        sector = "Unknown"
    sector_map.setdefault(sector, []).append(idx)

concepts = sorted(sector_map.keys())
stock2concept = np.zeros((len(instruments), len(concepts)), dtype=int)
for j, sector in enumerate(concepts):
    stock2concept[sector_map[sector], j] = 1

# 保存并同时把概念名称写入一个对照文件（可选）
np.save(base / "stock2concept.npy", stock2concept)
with open(base / "concept_names.txt", "w") as f:
    for sec in concepts:
        f.write(sec + "\n")

print(f"Saved stock2concept.npy with shape {stock2concept.shape}.")
print(f"Concept names written to concept_names.txt.")
