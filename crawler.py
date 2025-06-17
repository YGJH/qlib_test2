from qlib_stock_crawler import QlibStockCrawler
from datetime import datetime, timedelta  # 修復：正確導入 datetime
import os
def main():
    # 創建爬蟲
    crawler = QlibStockCrawler(output_dir="./data/csv_data")
    
    # 定義要爬取的股票 - 大幅擴展到800+支股票
    symbols = {
        "US": [
            # === 大型科技股 (FAANG+) ===
            "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "NVDA", "TSLA",
            "NFLX", "ADBE", "CRM", "ORCL", "INTC", "AMD", "QCOM", "AVGO",
            "PYPL", "SHOP", "UBER", "LYFT", "SNAP", "TWTR", "ZOOM", "DOCU",
            "PLTR", "SNOW", "COIN", "RBLX", "U", "NET", "DDOG", "CRWD",
            "ZM", "OKTA", "TWLO", "SPLK", "MDB", "TEAM", "HUBS", "ZEN",
            
            # === 雲端與軟體 ===
            "NOW", "WDAY", "VEEV", "INTU", "ADSK", "ANSS", "CDNS", "SNPS",
            "FTNT", "PANW", "ZS", "CYBR", "FEYE", "CHKP", "SAIL", "QLYS",
            "TENB", "RPD", "VRNS", "ESTC", "SUMO", "FROG", "AI",
            "C3AI", "PATH", "DOCN", "FSLY", "AKAM", "LUMN", "VZ",
            
            # === 半導體全產業鏈 ===
            "TSM", "ASML", "TXN", "ADI", "MRVL", "KLAC", "LRCX", "AMAT",
            "MU", "NXPI", "MCHP", "ON", "SWKS", "QRVO", "MPWR", "CRUS",
            "SYNA", "MXIM", "LATTICE", "SLAB", "SITM", "FORM",
            "SMCI", "WDC", "STX", "NTAP", "PURE", "HPQ", "DELL", "IBM",
            
            # === 金融服務 ===
            "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA",
            "COF", "USB", "PNC", "TFC", "BK", "STT", "SCHW", "SPGI", "ICE",
            "CME", "MCO", "NDAQ", "CBOE", "MKTX", "VIRT", "KKR", "BX", "APO",
            "CG", "OWL", "TPG", "ARES", "BLUE", "HLI", "PJT", "LAZ", "EVR",
            
            # === 保險 ===
            "BRK-A", "BRK-B", "AIG", "PGR", "TRV", "CB", "ALL",
            "MET", "PRU", "AFL", "GL", "AJG", "MMC", "AON", "WTW", "BRO",
            "RYAN", "ESGR", "JKHY", "FIS", "FISV", "SQ", "AFRM",
            
            # === 醫療保健與生技 ===
            "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY", "DHR",
            "BMY", "AMGN", "GILD", "VRTX", "REGN", "BIIB", "CVS", "ANTM",
            "CI", "HUM", "CNC", "MOH", "ISRG", "SYK", "MDT", "BSX", "ZBH",
            "BDX", "BAX", "EW", "HOLX", "DXCM", "ALGN", "ILMN", "INCY",
            "BMRN", "ALNY", "SRPT", "IONS", "MRNA", "BNTX", "NVAX", "VXRT",
            "CRSP", "EDIT", "NTLA", "BEAM", "FATE", "SGMO", "PACB",
            
            # === 消費品與零售 ===
            "WMT", "HD", "COST", "TGT", "LOW", "TJX", "DG", "DLTR", "ROST",
            "BBY", "GPS", "M", "JWN", "KSS", "BBBY", "BIG", "FIVE", "OLLI",
            "BURL", "URBN", "AEO", "ANF", "EXPR", "GES", "ZUMZ", "SHOO",
            "NKE", "LULU", "UAA", "UA", "DECK", "CROX", "SKX",
            "VFC", "HBI", "PVH", "RL", "CPRI", "TPR",
            
            # === 食品飲料 ===
            "PG", "KO", "PEP", "MCD", "SBUX", "YUM", "QSR",
            "DPZ", "CMG", "CHDN", "SHAK", "WING", "BLMN", "TXRH", "DRI",
            "EAT", "CAKE", "RUTH", "BJRI", "KRUS", "PLAY", "DAVE", "BROS",
            "KR", "SYY", "UNFI", "VLGEA", "CAG",
            "GIS", "K", "CPB", "SJM", "HRL", "TSN",
            
            # === 工業與航空 ===
            "BA", "CAT", "GE", "MMM", "HON", "RTX", "LMT", "NOC", "GD",
            "UPS", "FDX", "DAL", "AAL", "UAL", "LUV", "JBLU", "SAVE", "HA",
            "DE", "EMR", "ETN", "PH", "ITW", "ROK", "DOV", "IR", "AME",
            "FLS", "FLR", "JEC", "PWR", "BLDR", "DHI", "LEN", "NVR", "PHM",
            "TOL", "KBH", "MTH", "TMHC", "MHO", "LGIH", "GRBK", "TPH",
            
            # === 運輸與物流 ===
            "XPO", "ODFL", "CHRW", "EXPD", "HUBG", "LSTR",
            "ARCB", "JBHT", "KNX", "SAIA", "CVLG", "PTSI",
            "CSX", "UNP", "NSC", "KSU", "CP", "CNI", "GBX",
            
            # === 能源全產業鏈 ===
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "OXY",
            "BKR", "HAL", "DVN", "FANG", "APA", "MRO", "CLR", "AR", "SM",
            "CTRA", "OVV", "PR", "RRC", "EQT", "CNX", "ARCH", "BTU", "AMR",
            "CEIX", "HNRG", "METC", "SU", "TRP", "ENB", "KMI", "OKE",
            
            # === 公用事業 ===
            "NEE", "DUK", "SO", "D", "EXC", "XEL", "PEG", "SRE", "ED",
            "ETR", "WEC", "ES", "FE", "AEE", "CNP", "NI", "CMS", "DTE",
            "PPL", "AEP", "PCG", "EIX", "AWK", "WTR", "CTWS",
            
            # === 材料與化工 ===
            "LIN", "APD", "ECL", "SHW", "FCX", "NEM", "GOLD", "CF", "DOW",
            "DD", "PPG", "EMN", "IFF", "FMC", "LYB", "CE", "VMC", "MLM",
            "NUE", "STLD", "RS", "CMC", "X", "CLF", "MT", "TX", "VALE",
            "BHP", "RIO", "AA", "CENX", "KALU", "ATI", "HCC", "CRS",
            
            # === 房地產與REITs ===
            "AMT", "PLD", "CCI", "EQIX", "WELL", "DLR", "PSA", "EXR", "AVB",
            "EQR", "MAA", "ESS", "UDR", "CPT", "AIV", "O", "STAG", "IRM",
            "ARE", "BXP", "VTR", "PEAK", "DEI", "KIM", "REG", "FRT", "SPG",
            "MAC", "PEI", "CBL", "WPG", "TCO", "ROIC", "NNN", "ADC", "EPRT",
            
            # === 電動車與新能源 ===
            "F", "GM", "RIVN", "LCID", "NIO", "LI", "XPEV", "NKLA",
            "FSR", "GOEV", "RIDE", "WKHS", "HYLN", "SOLO", "ARVL", "CANOO",
            "ENVX", "QS", "CBAT", "PLUG", "FCEL",
            "BLDP", "BE", "CLNE", "GEVO", "KTOS", "EVGO", "CHPT", "BLNK",
            
            # === 主要大盤ETF ===
            "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "EEM", "EFA",
            "IEFA", "IEMG", "ACWI", "VT", "ITOT", "SCHB", "SCHA", "SCHF",
            "SCHE", "SCHX", "SCHY", "SCHZ", "IVV", "IVE", "IVW", "IJH",
            
            # === 行業ETF ===
            "XLF", "XLK", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB",
            "XLRE", "XLC", "SOXX", "XBI", "IBB", "KRE", "XHB", "XRT", "XME",
            "XOP", "JETS", "ICLN", "QCLN", "KWEB", "FXI", "ASHR", "MCHI",
            "VGK", "VPL", "VGT", "VHT", "VFH", "VIS", "VPU", "VDC", "VCR",
            
            # === 主題與創新ETF ===
            "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "BOTZ", "ROBO", "ESPO",
            "GAMR", "HACK", "CIBR", "SKYY", "FINX", "CLOU", "WCLD", "BUG",
            "HERO", "UFO", "MOON", "DRIV", "IDRV", "CARZ", "EKAR", "LIT",
            "BATT", "GRID", "SMOG", "PBW", "ERTH", "CTEC",
            
            # === 固定收益ETF ===
            "AGG", "BND", "TLT", "IEF", "SHY", "TIP", "HYG", "LQD", "EMB",
            "JNK", "VCIT", "VCSH", "VGIT", "VGSH", "VTEB", "MUB", "HYD",
            "SHYG", "SJNK", "FALN", "IGIB", "FLOT", "MINT", "NEAR", "GSY",
            
            # === 商品ETF ===
            "GLD", "SLV", "DJP", "DBA", "USO", "UNG", "PDBC", "IAU", "SIVR",
            "PALL", "PPLT", "URA", "REMX", "PICK", "COPX", "SIL", "GLTR",
            "BAR", "OUNZ", "SGOL", "PHYS", "PSLV", "AAAU", "GLDM", "IAUM",
            
            # === 國際與新興市場ETF ===
            "INDA", "EWJ", "EWZ", "EWG", "EWU", "EWC", "EWA", "EWY", "EWW",
            "EWH", "EWS", "EWT", "EWI", "EWP", "EWQ", "EWL", "EWK", "EWN",
            "EWD", "EWO", "EPOL", "EPP", "EZA", "ECH", "EPHE", "EIDO",
            
            # === 加密貨幣相關 ===
            "MSTR", "RIOT", "MARA", "HUT", "BITF", "CLSK", "CORZ",
            "HIVE", "ARBK", "CAN", "BTBT", "SOS", "EBANG", "GREE",
            
            # === 中概股 ===
            "BABA", "JD", "PDD", "BIDU", "TME", "IQ", "BILI", "VIPS", "WB",
            "DIDI", "TAL", "EDU", "YMM", "DOYU", "HUYA", "KC", "BZUN", "MOMO",
            "YY", "GRUB", "WUBA", "FENG", "SOHU", "SINA", "WEI", "TOUR",
            "TIGR", "FUTU", "UP", "QD", "TUYA", "API", "EHTH", "FINV",
            
            # === 娛樂媒體 ===
            "DIS", "CMCSA", "T", "TMUS", "CHTR", "DISH", "SIRI",
            "WBD", "PARA", "FOX", "FOXA", "LGF-A", "LGF-B", "MSGN", "NYT",
            "GOGO", "REZI", "IMAX", "CNK", "AMC", "NCMI", "RGS", "YELP",
            
            # === 生活消費服務 ===
            "EBAY", "ETSY", "W", "OSTK", "PRTS", "GRPN",
            "ANGI", "IAC", "MTCH", "BMBL", "PTON", "ROKU", "SPOT", "ZG", "Z",
            "ABNB", "EXPE", "BKNG", "TXG", "TRIP", "MMYT", "TCOM", "HTHT",
            
            # === 食品外送 ===
            "DASH", "BIRD", "LIME", "CPNG",
            
            # === 更多科技創新 ===
            "RGEN", "PACB", "ONT", "NTRA", "RXRX", "SDGR",
            "EXACT", "FGEN", "DNLI", "FOLD", "TWST",
        ]
    }
    
    # 去除重複股票
    symbols["US"] = list(set(symbols["US"]))  # 使用set去重
    symbols["US"].sort()  # 排序便於查看
    
    print(f"去重後總數: {len(symbols['US'])} 支股票和ETF")
    
    # 設定時間範圍
    start_date = "2019-01-01"  # 延長時間範圍獲取更多歷史數據
    end_date = "2024-12-31"
    
    print(f"開始爬取 {len(symbols['US'])} 支股票和ETF數據...")
    print(f"時間範圍: {start_date} 到 {end_date}")
    
    # 檢查上次運行時間
    last_run_file = "./data/last_run.txt"
    should_update = True
    
    try:
        if os.path.exists(last_run_file):
            with open(last_run_file, 'r') as f:
                date = f.read().strip()
                print(f"上次運行時間: {date}")
                # 修復：使用正確的 datetime.strptime
                last_dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                now = datetime.now()
                hours_diff = (now - last_dt).total_seconds() / 3600
                
                if hours_diff < 24:  # 24小時內不重複爬取
                    print(f"距離上次更新僅 {hours_diff:.1f} 小時，跳過爬取")
                    should_update = False
    except Exception as e:
        print(f"檢查上次運行時間時出錯: {e}")
        should_update = True
    
    if should_update:
        # 爬取美股
        print("\n=== 爬取美股 ===")
        us_data = crawler.get_us_stocks(symbols["US"], start_date, end_date)
        if us_data:
            print(f"成功獲取 {len(us_data)} 支股票數據")
            crawler.save_for_qlib(us_data)
            
            # 記錄運行時間
            os.makedirs("./data", exist_ok=True)
            with open(last_run_file, 'w') as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"\n數據已保存到: {crawler.output_dir}")
        print("現在可以使用 dump_bin.py 處理這些數據了!")
        


        # 運行 dump_bin.py 命令
        try:
            import subprocess
            PROC_DIR = str(crawler.output_dir)
            QLIB_DIR = os.path.expanduser(".qlib/qlib_data/my_us_data")
            INCLUDE_FIELDS = "open,close,high,low,volume"

            cmd = [
                "python",
                "scripts/dump_bin.py",
                "dump_all",
                "--csv_path", PROC_DIR,
                "--qlib_dir", QLIB_DIR,
                "--include_fields", INCLUDE_FIELDS,
            ]
            print("Running:", " ".join(cmd))
            subprocess.check_call(cmd)
            print("✓ dump_bin.py 執行完成")
        except Exception as e:
            print(f"執行 dump_bin.py 時出錯: {e}")
    else:
        print("跳過數據爬取，使用現有數據")

if __name__ == "__main__":
    main()