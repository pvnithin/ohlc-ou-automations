import pandas as pd
import numpy as np
import os
from datetime import datetime

# ================================
# CONFIGURATION
# ================================
base_folder = os.path.join(os.getcwd(), "data")
input_parquet_name = "daily_ohlc.parquet"
results_folder = "results_all"

reg_window = 60
ma_length = 200
min_theta = 0.02
max_half_life = 25

input_path = os.path.join(base_folder, input_parquet_name)
results_path = os.path.join(os.getcwd(), results_folder)
os.makedirs(results_path, exist_ok=True)

current_date = datetime.now().strftime('%Y-%m-%d')
output_filename = f"ou_signals_all_{current_date}.csv"
output_path = os.path.join(results_path, output_filename)

# ================================
# OU CALCULATION PER BAR
# ================================
def calculate_ou_parameters_at_bar(prices, bar_idx):
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
    sum_x = X_vals.sum()
    sum_y = dX_vals.sum()
    sum_xy = (X_vals * dX_vals).sum()
    sum_x2 = (X_vals * X_vals).sum()

    denom = n * sum_x2 - sum_x**2
    if denom == 0:
        return None

    slope_b = (n * sum_xy - sum_x * sum_y) / denom
    intercept_a = (sum_y - slope_b * sum_x) / n

    if slope_b >= 0 or slope_b <= -1:
        return None

    theta = -np.log(1 + slope_b)
    if theta <= 0:
        return None

    mu_ou = intercept_a / (1 - np.exp(-theta))

    predicted = intercept_a + slope_b * X_vals
    residuals = dX_vals - predicted
    sigma_ou = np.sqrt((residuals**2).sum() / (len(X_vals) - 2))

    eq_var = (sigma_ou**2) / (2 * theta)
    eq_stdev = np.sqrt(eq_var)

    band_upper = mu_ou + 2 * eq_stdev
    band_lower = mu_ou - 2 * eq_stdev
    half_life = np.log(2) / theta

    return {
        "theta": theta,
        "half_life": half_life,
        "Xt": Xt.iloc[-1],
        "band_upper": band_upper,
        "band_lower": band_lower
    }

# ================================
# SIGNAL DETECTOR
# ================================
def detect_signals(symbol_data):
    symbol_data = symbol_data.sort_values("Date").reset_index(drop=True)
    n = len(symbol_data)

    if n < ma_length + reg_window + 1:
        return []

    last = n - 1
    prev = n - 2

    prev_params = calculate_ou_parameters_at_bar(symbol_data["Close"], prev)
    curr_params = calculate_ou_parameters_at_bar(symbol_data["Close"], last)

    if prev_params is None or curr_params is None:
        return []

    theta = curr_params["theta"]
    half_life = curr_params["half_life"]

    Xt_prev = prev_params["Xt"]
    Xt_curr = curr_params["Xt"]

    lower_prev = prev_params["band_lower"]
    upper_prev = prev_params["band_upper"]

    lower_curr = curr_params["band_lower"]
    upper_curr = curr_params["band_upper"]

    regime_ok = (theta >= min_theta) and (half_life <= max_half_life)

    raw_buy = (Xt_prev >= lower_prev) and (Xt_curr < lower_curr) and (Xt_curr < Xt_prev)
    raw_sell = (Xt_prev <= upper_prev) and (Xt_curr > upper_curr) and (Xt_curr > Xt_prev)

    buy_signal = raw_buy and regime_ok
    sell_signal = raw_sell and regime_ok

    date = symbol_data["Date"].iloc[last]
    close = symbol_data["Close"].iloc[last]

    signals = []
    if buy_signal or sell_signal:
        signals.append({
            "Signal": "BUY" if buy_signal else "SELL",
            "Date": date,
            "Close": close,
            "Theta": theta,
            "Half_Life": half_life,
            "Deviation_Xt": Xt_curr,
            "Prev_Xt": Xt_prev,
            "Upper_Band": upper_curr,
            "Lower_Band": lower_curr,
            "Regime_Valid": True
        })
    elif raw_buy or raw_sell:
        signals.append({
            "Signal": "FILTERED_BUY" if raw_buy else "FILTERED_SELL",
            "Date": date,
            "Close": close,
            "Theta": theta,
            "Half_Life": half_life,
            "Deviation_Xt": Xt_curr,
            "Prev_Xt": Xt_prev,
            "Upper_Band": upper_curr,
            "Lower_Band": lower_curr,
            "Regime_Valid": False
        })

    return signals

# ================================
# MAIN EXECUTION
# ================================
print("OU Signal Scanner (ALL STOCKS)")
print("Loading data...")

df = pd.read_parquet(input_path)
df["Date"] = pd.to_datetime(df["Date"])

symbols = df["Symbol"].unique()

results = []

for s in symbols:
    sub = df[df["Symbol"] == s].copy()
    sigs = detect_signals(sub)
    for sig in sigs:
        sig["Symbol"] = s
        results.append(sig)

# ==============================================
# APPLY FILTERS BEFORE SAVING (YOUR REQUEST)
# ==============================================

if results:
    out = pd.DataFrame(results)

    out = out[
        (out["Close"] > 50) &
        (out["Regime_Valid"] == True) &
        (out["Signal"] == "BUY")
    ]

    out = out.sort_values("Date", ascending=False)

    out.to_csv(output_path, index=False)
    print(f"Saved filtered results → {output_filename}")
else:
    print("No signals found.")
