import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
base_folder = "scripts"          # GitHub folder
input_parquet_name = "data/daily_ohlc.parquet"
results_folder = "results_all"

# OU Process Parameters
reg_window = 60
ma_length = 200
min_theta = 0.02
max_half_life = 25

# --- End Configuration ---

input_path = os.path.join(base_folder, input_parquet_name)

results_path = os.path.join(base_folder, results_folder)
os.makedirs(results_path, exist_ok=True)

current_date = datetime.now().strftime('%Y-%m-%d')
output_filename = f"ou_signals_all_{current_date}.csv"
output_path = os.path.join(results_path, output_filename)


# -------------------------------------------------------------------------
# OU CALCULATIONS
# -------------------------------------------------------------------------

def calculate_ou_parameters_at_bar(prices, reg_window, ma_length, bar_idx):

    if bar_idx < ma_length + reg_window:
        return None

    prices_subset = prices.iloc[:bar_idx + 1]

    long_term_mean = prices_subset.rolling(window=ma_length).mean()

    Xt = prices_subset - long_term_mean
    dX = Xt.diff()
    X_prev = Xt.shift(1)

    last_idx = len(prices_subset) - 1
    start_idx = last_idx - reg_window + 1

    if start_idx < ma_length + 1:
        return None

    X_vals = X_prev.iloc[start_idx:last_idx + 1].values
    dX_vals = dX.iloc[start_idx:last_idx + 1].values

    valid_mask = ~(np.isnan(X_vals) | np.isnan(dX_vals))
    X_vals = X_vals[valid_mask]
    dX_vals = dX_vals[valid_mask]

    if len(X_vals) < reg_window * 0.8:
        return None

    n = len(X_vals)
    sum_x = np.sum(X_vals)
    sum_y = np.sum(dX_vals)
    sum_xy = np.sum(X_vals * dX_vals)
    sum_x2 = np.sum(X_vals ** 2)

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return None

    slope_b = (n * sum_xy - sum_x * sum_y) / denominator
    intercept_a = (sum_y - slope_b * sum_x) / n

    if slope_b >= 0 or slope_b <= -1:
        return None

    theta = -np.log(1 + slope_b)
    if theta <= 0:
        return None

    mu_ou = intercept_a / (1 - np.exp(-theta))

    predicted_y = intercept_a + slope_b * X_vals
    residuals = dX_vals - predicted_y
    sum_residuals_sq = np.sum(residuals ** 2)
    sigma_ou = np.sqrt(sum_residuals_sq / (len(X_vals) - 2))

    eq_variance = (sigma_ou ** 2) / (2 * theta)
    eq_stdev = np.sqrt(eq_variance)

    band_upper = mu_ou + (2.0 * eq_stdev)
    band_lower = mu_ou - (2.0 * eq_stdev)

    half_life = np.log(2) / theta if theta > 0 else np.inf
    Xt_value = Xt.iloc[-1]

    return {
        'theta': theta,
        'mu_ou': mu_ou,
        'sigma_ou': sigma_ou,
        'half_life': half_life,
        'Xt': Xt_value,
        'band_upper': band_upper,
        'band_lower': band_lower
    }


# -------------------------------------------------------------------------
# SIGNAL DETECTION
# -------------------------------------------------------------------------

def detect_signals(symbol_data):

    results = []

    symbol_data = symbol_data.sort_values("Date").reset_index(drop=True)

    if len(symbol_data) < ma_length + reg_window + 1:
        return results

    last_bar_idx = len(symbol_data) - 1
    prev_bar_idx = last_bar_idx - 1

    current_params = calculate_ou_parameters_at_bar(
        symbol_data["Close"], reg_window, ma_length, last_bar_idx
    )
    prev_params = calculate_ou_parameters_at_bar(
        symbol_data["Close"], reg_window, ma_length, prev_bar_idx
    )

    if current_params is None or prev_params is None:
        return results

    theta = current_params["theta"]
    half_life = current_params["half_life"]
    regime_valid = (theta >= min_theta) and (half_life <= max_half_life)

    prev_Xt = prev_params["Xt"]
    curr_Xt = current_params["Xt"]

    prev_lower = prev_params["band_lower"]
    prev_upper = prev_params["band_upper"]
    curr_lower = current_params["band_lower"]
    curr_upper = current_params["band_upper"]

    raw_buy = (prev_Xt >= prev_lower) and (curr_Xt < curr_lower) and (curr_Xt < prev_Xt)
    raw_sell = (prev_Xt <= prev_upper) and (curr_Xt > curr_upper) and (curr_Xt > prev_Xt)

    buy_signal = raw_buy and regime_valid
    sell_signal = raw_sell and regime_valid

    signal_date = symbol_data["Date"].iloc[last_bar_idx]
    latest_close = symbol_data["Close"].iloc[last_bar_idx]

    if buy_signal or sell_signal:
        signal_type = "BUY" if buy_signal else "SELL"
    else:
        return results

    results.append({
        "Signal": signal_type,
        "Theta": theta,
        "Half_Life": half_life,
        "Deviation_Xt": curr_Xt,
        "Prev_Xt": prev_Xt,
        "Upper_Band": curr_upper,
        "Lower_Band": curr_lower,
        "Regime_Valid": regime_valid,
        "Date": signal_date,
        "Close": latest_close
    })

    return results


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

print(f"OU Process Signal Scanner - ALL STOCKS")
print(f"Date: {current_date}")

df = pd.read_parquet(input_path)
df["Date"] = pd.to_datetime(df["Date"])

symbols = df["Symbol"].unique()

all_signals = []

for symbol in symbols:
    symbol_data = df[df["Symbol"] == symbol].copy()
    signals = detect_signals(symbol_data)

    for s in signals:
        s["Symbol"] = symbol
        all_signals.append(s)

# Convert to DF
results_df = pd.DataFrame(all_signals)

# -------------------------------------------------------------------------
# ✅ APPLY THE FILTER YOU REQUESTED
# -------------------------------------------------------------------------

results_df = results_df[
    (results_df["Close"] > 50) &
    (results_df["Regime_Valid"] == True) &
    (results_df["Signal"] == "BUY")
]

# -------------------------------------------------------------------------

results_df = results_df.sort_values("Date", ascending=False)
results_df.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Done.")
