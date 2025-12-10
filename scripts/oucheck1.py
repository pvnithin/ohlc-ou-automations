import pandas as pd
import numpy as np
import os
from datetime import datetime

# Fixed parquet path (correct location)
input_path = "../data/daily_ohlc.parquet"

results_folder = "results_selected"
output_base = "scripts"

# OU params
reg_window = 60
ma_length = 200
min_theta = 0.02
max_half_life = 25

os.makedirs(os.path.join(output_base, results_folder), exist_ok=True)

current_date = datetime.now().strftime("%Y-%m-%d")
output_filename = f"ou_signals_{current_date}.csv"
output_path = os.path.join(output_base, results_folder, output_filename)

# SYMBOL LIST (from your 2.py)
SYMBOLS_TO_SCAN = [
    '360ONE', 'ABB', 'APLAPOLLO', 'AUBANK', 'ADANIENSOL', 'ADANIENT', 'ADANIGREEN', 'ADANIPORTS',
    'ABCAPITAL', 'ALKEM', 'AMBER', 'AMBUJACEM', 'ANGELONE', 'APOLLOHOSP', 'ASHOKLEY', 'ASIANPAINT',
    'ASTRAL', 'AUROPHARMA', 'DMART', 'AXISBANK', 'BSE', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV',
    'BANKBARODA', 'BANKINDIA', 'BDL', 'BEL', 'BHARATFORG', 'BHEL', 'BPCL', 'BHARTIARTL',
    'BIOCON', 'BLUESTARCO', 'BOSCHLTD', 'BRITANNIA', 'CGPOWER', 'CANBK', 'CDSL', 'CHOLAFIN',
    'CIPLA', 'COALINDIA', 'COFORGE', 'COLPAL', 'CAMS', 'CONCOR', 'CROMPTON', 'CUMMINSIND',
    'CYIENT', 'DLF', 'DABUR', 'DALBHARAT', 'DELHIVERY', 'DIVISLAB', 'DIXON', 'DRREDDY',
    'ETERNAL', 'EICHERMOT', 'EXIDEIND', 'NYKAA', 'FORTIS', 'GAIL', 'GMRAIRPORT', 'GLENMARK',
    'GODREJCP', 'GODREJPROP', 'GRASIM', 'HCLTECH', 'HDFCAMC', 'HDFCBANK', 'HDFCLIFE', 'HFCL',
    'HAVELLS', 'HEROMOTOCO', 'HINDALCO', 'HAL', 'HINDPETRO', 'HINDUNILVR', 'HINDZINC', 'HUDCO',
    'ICICIBANK', 'ICICIGI', 'ICICIPRULI', 'IDFCFIRSTB', 'IIFL', 'ITC', 'INDIANB', 'IEX',
    'IOC', 'IRCTC', 'IRFC', 'IREDA', 'IGL', 'INDUSTOWER', 'INDUSINDBK', 'NAUKRI', 'INFY',
    'INOXWIND', 'INDIGO', 'JSWENERGY', 'JSWSTEEL', 'JINDALSTEL', 'JIOFIN', 'JUBLFOOD', 'KEI',
    'KPITTECH', 'KALYANKJIL', 'KAYNES', 'KFINTECH', 'KOTAKBANK', 'LTF', 'LICHSGFIN', 'LTIM',
    'LT', 'LAURUSLABS', 'LICI', 'LODHA', 'LUPIN', 'M&M', 'MANAPPURAM', 'MANKIND', 'MARICO',
    'MARUTI', 'MFSL', 'MAXHEALTH', 'MAZDOCK', 'MPHASIS', 'MCX', 'MUTHOOTFIN', 'NBCC', 'NCC',
    'NHPC', 'NMDC', 'NTPC', 'NATIONALUM', 'NESTLEIND', 'NUVAMA', 'OBEROIRLTY', 'ONGC', 'OIL',
    'PAYTM', 'OFSS', 'POLICYBZR', 'PGEL', 'PIIND', 'PNBHOUSING', 'PAGEIND', 'PATANJALI',
    'PERSISTENT', 'PETRONET', 'PIDILITIND', 'PPLPHARMA', 'POLYCAB', 'PFC', 'POWERGRID',
    'PRESTIGE', 'PNB', 'RBLBANK', 'RECLTD', 'RVNL', 'RELIANCE', 'SBICARD', 'SBILIFE',
    'SHREECEM', 'SRF', 'SAMMAANCAP', 'MOTHERSON', 'SHRIRAMFIN', 'SIEMENS', 'SOLARINDS',
    'SONACOMS', 'SBIN', 'SAIL', 'SUNPHARMA', 'SUPREMEIND', 'SUZLON', 'SYNGENE', 'TATACONSUM',
    'TITAGARH', 'TVSMOTOR', 'TATACHEM', 'TCS', 'TATAELXSI', 'TATAMOTORS', 'TATAPOWER',
    'TATASTEEL', 'TATATECH', 'TECHM', 'FEDERALBNK', 'INDHOTEL', 'PHOENIXLTD', 'TITAN',
    'TORNTPHARM', 'TORNTPOWER', 'TRENT', 'TIINDIA', 'UNOMINDA', 'UPL', 'ULTRACEMCO',
    'UNIONBANK', 'UNITDSPR', 'VBL', 'VEDL', 'IDEA', 'VOLTAS', 'WIPRO', 'YESBANK', 'ZYDUSLIFE', 'BANDHANBNK'
]

# OU calculation function
def calculate_ou_parameters_at_bar(prices, reg_window, ma_length, bar_idx):

    if bar_idx < ma_length + reg_window:
        return None

    prices_subset = prices.iloc[:bar_idx + 1]
    long_mean = prices_subset.rolling(ma_length).mean()

    Xt = prices_subset - long_mean
    dX = Xt.diff()
    X_prev = Xt.shift(1)

    last = len(prices_subset) - 1
    start = last - reg_window + 1

    if start < ma_length + 1:
        return None

    X_vals = X_prev.iloc[start:last + 1].values
    dX_vals = dX.iloc[start:last + 1].values

    valid = ~(np.isnan(X_vals) | np.isnan(dX_vals))
    X_vals = X_vals[valid]
    dX_vals = dX_vals[valid]

    if len(X_vals) < reg_window * 0.8:
        return None

    n = len(X_vals)
    sx = X_vals.sum()
    sy = dX_vals.sum()
    sxy = (X_vals * dX_vals).sum()
    sx2 = (X_vals ** 2).sum()

    denom = n * sx2 - sx*sx
    if denom == 0:
        return None

    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n

    if b >= 0 or b <= -1:
        return None

    theta = -np.log(1 + b)
    if theta <= 0:
        return None

    mu = a / (1 - np.exp(-theta))

    pred = a + b * X_vals
    resid = dX_vals - pred
    sigma = np.sqrt((resid**2).sum() / (len(X_vals) - 2))

    eq_var = sigma**2 / (2 * theta)
    eq_sd = np.sqrt(eq_var)

    upper = mu + 2 * eq_sd
    lower = mu - 2 * eq_sd
    half_life = np.log(2) / theta

    return {"theta": theta, "half_life": half_life, "Xt": Xt.iloc[-1], "upper": upper, "lower": lower}


# Signal detection
def detect_signals(df_symbol):

    df_symbol = df_symbol.sort_values("Date").reset_index(drop=True)

    if len(df_symbol) < ma_length + reg_window + 1:
        return []

    last = len(df_symbol) - 1
    prev = last - 1

    prev_params = calculate_ou_parameters_at_bar(df_symbol["Close"], reg_window, ma_length, prev)
    curr_params = calculate_ou_parameters_at_bar(df_symbol["Close"], reg_window, ma_length, last)

    if prev_params is None or curr_params is None:
        return []

    theta = curr_params["theta"]
    hl = curr_params["half_life"]
    regime_ok = (theta >= min_theta and hl <= max_half_life)

    Xp = prev_params["Xt"]
    Xc = curr_params["Xt"]

    lp = prev_params["lower"]
    up = prev_params["upper"]

    lc = curr_params_
