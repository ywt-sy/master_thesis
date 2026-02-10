import pandas as pd
import numpy as np
import os

def load_returns_data(n,k):
    if k == 5:
        file_paths = {
            'AAPL': 'data/real/data_stocks/aapl_us_w.csv',
            'Coca': 'data/real/data_stocks/ko_us_w.csv',
            'JPM': 'data/real/data_stocks/jpm_us_w.csv',
            'UHealth': 'data/real/data_stocks/unh_us_w.csv',
            'XOM': 'data/real/data_stocks/xom_us_w.csv'
        }
    
    if k == 10:
        file_paths = {
        'AAPL': 'data/real/data_stocks/aapl_us_w.csv',
        'XOM': 'data/real/data_stocks/xom_us_w.csv',
        'KDDI': 'data/real/data_stocks/9433_jp_w.csv',
        'JPM': 'data/real/data_stocks/jpm_us_w.csv',
        'Coca': 'data/real/data_stocks/ko_us_w.csv',
        'Toyota': 'data/real/data_stocks/7203_jp_w.csv',
        'UHealth': 'data/real/data_stocks/unh_us_w.csv',
        'Mitsubishi Electric': 'data/real/data_stocks/6503_jp_w.csv',
        'Nflx': 'data/real/data_stocks/nflx_us_w.csv',
        'DOW': 'data/real/data_stocks/dow_us_w.csv',
        }

    if k == 15:
        file_paths = {
        'AAPL': 'data/real/data_stocks/aapl_us_w.csv',
        'XOM': 'data/real/data_stocks/xom_us_w.csv',
        'JPM': 'data/real/data_stocks/jpm_us_w.csv',
        'Coca': 'data/real/data_stocks/ko_us_w.csv',
        'Toyota': 'data/real/data_stocks/7203_jp_w.csv',
        'UHealth': 'data/real/data_stocks/unh_us_w.csv',
        'Mitsubishi Electric': 'data/real/data_stocks/6503_jp_w.csv',
        'Nflx': 'data/real/data_stocks/nflx_us_w.csv',
        'DOW': 'data/real/data_stocks/dow_us_w.csv',
        #'Adobe': 'data/real/data_stocks/adbe_us_w.csv',
        'TTE': 'data/real/data_stocks/tte_us_w.csv',
        'BHP': 'data/real/data_stocks/bhp_uk_w.csv',
        'MSFT': 'data/real/data_stocks/msft_us_w.csv',
        #'Taketa': 'data/real/data_stocks/4502_jp_w.csv',
        #'NMR': 'data/real/data_stocks/8604_jp_w.csv',
        'Kao': 'data/real/data_stocks/4452_jp_w.csv',
        'KDDI': 'data/real/data_stocks/9433_jp_w.csv',
        #'UFJ': 'data/real/data_stocks/8306_jp_w.csv',
        #'Disney': 'data/real/data_stocks/dis_us_w.csv',
        #'Google': 'data/real/data_stocks/googl_us_w.csv',
        'SONY': 'data/real/data_stocks/sony_us_w.csv',
        #'Tsla': 'data/real/data_stocks/tsla_us_w.csv',
        }

    os.makedirs(f"data/real", exist_ok=True)  # Creates the folder if it doesn't exist

    returns_data = {}
    for asset, path in file_paths.items():
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        df['Return'] = df['Close'].pct_change()
        returns_data[asset] = df['Return'].dropna().tail(n)
    # Save
    returns_data = pd.DataFrame(returns_data)
    #returns_data.to_pickle("data/real/returns.pkl")  # ← This preserves structure!
    np.save(f"data/real/{n}_{k}_returns.npy", returns_data)

    return returns_data

# === Save as .dat for TikZ ===
def save_returns_as_dat(df, output_dir="data/real/stock_times/tikz_data"):
    os.makedirs(output_dir, exist_ok=True)

    for col in df.columns:
        with open(f"{output_dir}/{col.lower()}_returns.dat", "w") as f:
            f.write("x y\n")
            for i, val in enumerate(df[col]):
                f.write(f"{i} {val:.6f}\n")

# === Run ===
if __name__ == "__main__":
    n = 104
    k = 10
    returns = load_returns_data(n,k)
    save_returns_as_dat(returns)