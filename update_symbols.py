import yfinance as yf
import pandas as pd
from pathlib import Path

def test_and_update_symbols():
    """測試美股代號是否可用，並更新 symbols.txt"""
    
    # 候選美股代號列表（科技股為主）
    candidate_symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", 
        "ADBE", "CRM", "PYPL", "INTC", "AMD", "QCOM", "AVGO", "TXN",
        "ORCL", "NOW", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "MCHP",
        "ADI", "FTNT", "PANW", "CRWD", "ZS", "DDOG", "COST", "PEP",
        "CSCO", "CMCSA", "PDD", "TMUS", "ASML", "ADSK", "ROP", "PAYX",
        "FAST", "CTSH", "ODFL", "TEAM", "VRSK", "EXC", "LULU", "DXCM",
        "IDXX", "CTAS", "FANG", "BIIB", "ILMN", "KDP", "BKNG", "SBUX"
    ]
    
    print(f"測試 {len(candidate_symbols)} 個美股代號...")
    
    valid_symbols = []
    invalid_symbols = []
    
    for i, symbol in enumerate(candidate_symbols, 1):
        print(f"[{i}/{len(candidate_symbols)}] 測試 {symbol}...", end=" ")
        
        try:
            ticker = yf.Ticker(symbol)
            # 測試下載最近3個月的數據
            data = ticker.history(period="3mo")
            
            if not data.empty and len(data) > 10:  # 至少要有10個交易日的數據
                valid_symbols.append(symbol)
                print("✓")
            else:
                invalid_symbols.append(symbol)
                print("✗ (無數據)")
                
        except Exception as e:
            invalid_symbols.append(symbol)
            print(f"✗ (錯誤: {str(e)[:30]})")
    
    print(f"\n結果總結:")
    print(f"✓ 有效代號: {len(valid_symbols)}")
    print(f"✗ 無效代號: {len(invalid_symbols)}")
    
    if valid_symbols:
        # 寫入 symbols.txt
        symbols_file = Path("symbols.txt")
        with open(symbols_file, "w", encoding="utf-8") as f:
            for symbol in valid_symbols:
                f.write(f"{symbol}\n")
        
        print(f"\n✓ 已將 {len(valid_symbols)} 個有效代號寫入 symbols.txt")
        print("有效的美股代號:")
        for i, symbol in enumerate(valid_symbols, 1):
            if i % 10 == 1:  # 每10個換行
                print()
            print(f"{symbol:6}", end="")
        print()
        
        if invalid_symbols:
            print(f"\n無效的代號: {invalid_symbols}")
    else:
        print("\n✗ 沒有找到任何有效的美股代號！")
    
    return valid_symbols

if __name__ == "__main__":
    valid_symbols = test_and_update_symbols()
    print(f"\n完成！找到 {len(valid_symbols)} 個可用的美股代號。")