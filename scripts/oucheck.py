import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Load Parquet (fixed path) ---
input_path = "../data/daily_ohlc.parquet"

results_folder = "results_all"
output_base = "scripts"

reg_window = 60
ma_length = 200
min_theta = 0.02
max_half_life = 25

os.makedirs(os.path.join(output_base, results_folder), exist_ok=True)

current_date = datetime.now().strftime("%Y-%m-%d")
output_filename = f"ou_signals_all_{current_date}.csv"
output_path = os.path.join(output_base, results_folder, output_filename)


# -------------------------------------------------------------------------
# OU FUNCTION
# -------------------------------------------------------------------------
def calculate_ou_parameters_at_bar(prices, reg_window, ma_length, bar_idx):

    if bar_idx < ma_length + reg_window:
        return None

    prices_subset = prices.iloc[:bar_idx + 1]
    long_term_mean = prices_subset.rolling(ma_length).mean()

    Xt = prices_subset - long_term_mean
    dX = Xt.diff()
    X_prev = Xt.shift(1)

    last_idx = len(prices_subset) - 1
    start_idx = last_idx - reg_window + 1

    if start_idx < ma_length + 1:
        return None

    X_vals = X_prev.iloc[start_idx:last_idx + 1].values
    dX_vals = dX.iloc[start_idx:last_idx + 1].values

    valid = ~(np.isnan(X_vals) | np.isnan(dX_vals))
    X_vals = X_vals[valid]
    dX_vals = dX_vals[valid]

    if len(X_vals) < reg_window * 0.8:
        return None

    n = len(X_vals)
    sx = X_vals.sum()
    sy = dX_vals.sum()
    sxy = (X_vals * dX_vals).sum()
    sx2 = (X_vals**2).sum()

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

    return {
        "theta": theta,
        "half_life": half_life,
        "Xt": Xt.iloc[-1],
        "upper": upper,
        "lower": lower
    }


# -------------------------------------------------------------------------
# SIGNAL DETECTION
# -------------------------------------------------------------------------
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
    half_life = curr_params["half_life"]
    regime_ok = (theta >= min_theta and half_life <= max_half_life)

    Xp = prev_params["Xt"]
    Xc = curr_params["Xt"]

    lp = prev_params["lower"]
    up = prev_params["upper"]

    lc = curr_params["lower"]
    uc = curr_params["upper"]

    raw_buy = (Xp >= lp) and (Xc < lc) and (Xc < Xp)
    raw_sell = (Xp <= up) and (Xc > uc) and (Xc > Xp)

    date = df_symbol["Date"].iloc[last]
    close = df_symbol["Close"].iloc[last]

    if not (raw_buy or raw_sell):
        return []

    signal_type = "BUY" if raw_buy else "SELL"

    return [{
        "Symbol": df_symbol["Symbol"].iloc[0],
        "Date": date,
        "Close": close,
        "Signal": signal_type,
        "Theta": theta,
        "Half_Life": half_life,
        "Deviation_Xt": Xc,
        "Prev_Xt": Xp,
        "Upper_Band": uc,
        "Lower_Band": lc,
        "Regime_Valid": regime_ok
    }]


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
print("OU Process Signal Scanner - ALL STOCKS")

df = pd.read_parquet(input_path)
df["Date"] = pd.to_datetime(df["Date"])

symbols = df["Symbol"].unique()

signals = []

for sym in symbols:
    sub = df[df["Symbol"] == sym].copy()
    found = detect_signals(sub)
    for f in found:
        signals.append(f)

df_out = pd.DataFrame(signals)

# FILTERS REQUESTED BY YOU
df_out = df_out[
    (df_out["Close"] > 50) &
    (df_out["Regime_Valid"] == True) &
    (df_out["Signal"] == "BUY")
]

df_out = df_out.sort_values("Date", ascending=False)
df_out.to_csv(output_path, index=False)

print("Saved:", output_path)
