import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
# Use relative paths from scripts folder
script_dir = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_dir)  # Parent directory (repo root)
data_folder = os.path.join(base_folder, "data")
results_folder = os.path.join(base_folder, "results_drift_dominance")

input_parquet_name = "daily_ohlc.parquet"

# OU Process Parameters (matching Pine Script)
reg_window = 60  # Estimation Lookback Window
ma_length = 200  # Long-term Mean Length
min_theta = 0.02  # Min Theta for Signal
max_half_life = 25  # Max Half-Life for Signal
z0 = 1.28  # Drift dominance threshold (80% confidence)

# --- End Configuration ---

input_path = os.path.join(data_folder, input_parquet_name)

# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Generate filename with current date
current_date = datetime.now().strftime('%Y-%m-%d')
output_filename = f"ou_signals_drift_{current_date}.csv"
output_path = os.path.join(results_folder, output_filename)

def calculate_ou_parameters_at_bar(prices, reg_window, ma_length, bar_idx):
    """
    Calculate OU parameters AS THEY WERE at a specific bar
    This ensures we're comparing Xt to the bands that existed at that time
    """
    if bar_idx < ma_length + reg_window:
        return None
    
    # Use data up to and including bar_idx
    prices_subset = prices.iloc[:bar_idx + 1]
    
    # Calculate long-term mean
    long_term_mean = prices_subset.rolling(window=ma_length).mean()
    
    # Deviation from mean
    Xt = prices_subset - long_term_mean
    
    # Change in deviation
    dX = Xt.diff()
    X_prev = Xt.shift(1)
    
    # Get the last reg_window points for regression (up to bar_idx)
    last_idx = len(prices_subset) - 1
    start_idx = last_idx - reg_window + 1
    
    if start_idx < ma_length + 1:
        return None
    
    X_vals = X_prev.iloc[start_idx:last_idx + 1].values
    dX_vals = dX.iloc[start_idx:last_idx + 1].values
    
    # Remove NaN values
    valid_mask = ~(np.isnan(X_vals) | np.isnan(dX_vals))
    X_vals = X_vals[valid_mask]
    dX_vals = dX_vals[valid_mask]
    
    if len(X_vals) < reg_window * 0.8:  # Need at least 80% valid data
        return None
    
    # Linear regression: dX = a + b * X_prev
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
    
    # Extract OU parameters
    if slope_b >= 0 or slope_b <= -1:
        return None
    
    theta = -np.log(1 + slope_b)
    
    if theta <= 0:
        return None
    
    mu_ou = intercept_a / (1 - np.exp(-theta))
    
    # Calculate sigma
    predicted_y = intercept_a + slope_b * X_vals
    residuals = dX_vals - predicted_y
    sum_residuals_sq = np.sum(residuals ** 2)
    sigma_ou = np.sqrt(sum_residuals_sq / (len(X_vals) - 2))
    
    # Calculate bands
    eq_variance = (sigma_ou ** 2) / (2 * theta)
    eq_stdev = np.sqrt(eq_variance)
    band_upper = mu_ou + (2.0 * eq_stdev)
    band_lower = mu_ou - (2.0 * eq_stdev)
    
    # Half-life
    half_life = np.log(2) / theta if theta > 0 else np.inf
    
    # Get Xt at this bar
    Xt_value = Xt.iloc[-1]
    
    # Calculate drift z-score: (theta * |Xt - mu|) / sigma
    drift_z = (theta * np.abs(Xt_value - mu_ou)) / sigma_ou if sigma_ou > 0 else 0.0
    
    return {
        'theta': theta,
        'mu_ou': mu_ou,
        'sigma_ou': sigma_ou,
        'half_life': half_life,
        'Xt': Xt_value,
        'band_upper': band_upper,
        'band_lower': band_lower,
        'drift_z': drift_z
    }

def detect_signals(symbol_data):
    """
    Detect buy/sell signals using drift dominance logic:
    - Buy when: Xt < lower band AND regime valid AND drift dominant
    - Sell when: Xt > upper band AND regime valid AND drift dominant
    
    This captures the moment when mean-reversion force overcomes noise,
    signaling an imminent reversal BEFORE it happens.
    """
    results = []
    
    # Sort by date
    symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
    
    if len(symbol_data) < ma_length + reg_window + 1:
        return results
    
    # Get the last bar index
    last_bar_idx = len(symbol_data) - 1
    
    # Calculate OU parameters for the last bar
    current_params = calculate_ou_parameters_at_bar(
        symbol_data['Close'], reg_window, ma_length, last_bar_idx
    )
    
    if current_params is None:
        return results
    
    # Extract parameters
    theta = current_params['theta']
    half_life = current_params['half_life']
    current_Xt = current_params['Xt']
    current_band_lower = current_params['band_lower']
    current_band_upper = current_params['band_upper']
    drift_z = current_params['drift_z']
    mu_ou = current_params['mu_ou']
    sigma_ou = current_params['sigma_ou']
    
    # Three-factor signal logic:
    # 1. Regime Valid: theta >= min_theta AND half_life <= max_half_life
    # 2. At Extreme: Xt outside 2σ bands
    # 3. Drift Dominant: drift z-score > z0 (mean-reversion force > noise)
    regime_valid = (theta >= min_theta) and (half_life <= max_half_life)
    drift_dominant = drift_z > z0
    
    # Extremity checks
    at_lower_extreme = current_Xt < current_band_lower
    at_upper_extreme = current_Xt > current_band_upper
    
    # Final signals
    buy_signal = at_lower_extreme and regime_valid and drift_dominant
    sell_signal = at_upper_extreme and regime_valid and drift_dominant
    
    # Signal date is the current bar date
    signal_date = symbol_data['Date'].iloc[last_bar_idx]
    latest_close = symbol_data['Close'].iloc[last_bar_idx]
    
    # Record signal or filtered signal
    if buy_signal or sell_signal:
        signal_type = 'BUY' if buy_signal else 'SELL'
        results.append({
            'Signal': signal_type,
            'Theta': theta,
            'Half_Life': half_life,
            'Deviation_Xt': current_Xt,
            'Mu_OU': mu_ou,
            'Sigma_OU': sigma_ou,
            'Upper_Band': current_band_upper,
            'Lower_Band': current_band_lower,
            'Drift_Z': drift_z,
            'Regime_Valid': regime_valid,
            'Drift_Dominant': drift_dominant,
            'Date': signal_date,
            'Close': latest_close
        })
    elif at_lower_extreme or at_upper_extreme:
        # Filtered signal - at extreme but missing other conditions
        signal_type = 'FILTERED_BUY' if at_lower_extreme else 'FILTERED_SELL'
        filter_reasons = []
        
        if not regime_valid:
            if theta < min_theta:
                filter_reasons.append(f"Theta={theta:.4f}<{min_theta}")
            if half_life > max_half_life:
                filter_reasons.append(f"HalfLife={half_life:.2f}>{max_half_life}")
        
        if not drift_dominant:
            filter_reasons.append(f"DriftZ={drift_z:.2f}<={z0}")
        
        results.append({
            'Signal': signal_type,
            'Theta': theta,
            'Half_Life': half_life,
            'Deviation_Xt': current_Xt,
            'Mu_OU': mu_ou,
            'Sigma_OU': sigma_ou,
            'Upper_Band': current_band_upper,
            'Lower_Band': current_band_lower,
            'Drift_Z': drift_z,
            'Regime_Valid': regime_valid,
            'Drift_Dominant': drift_dominant,
            'Filter_Reason': '; '.join(filter_reasons),
            'Date': signal_date,
            'Close': latest_close
        })
    
    return results

# --- Main Execution ---
print(f"OU Process Signal Scanner - DRIFT DOMINANCE LOGIC")
print(f"Scan Date: {current_date}")
print(f"Results will be saved to: results_drift_dominance/{output_filename}")
print("=" * 70)
print(f"\nLoading OHLC data from {input_parquet_name}...")

try:
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} records for {df['Symbol'].nunique()} symbols")
except FileNotFoundError:
    print(f"Error: File not found at {input_path}")
    exit()
except Exception as e:
    print(f"Error reading parquet file: {e}")
    exit()

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Get list of all symbols
symbols = df['Symbol'].unique()
print(f"\nScanning {len(symbols)} symbols for OU signals...")
print(f"\nSignal Logic (3 factors required):")
print(f"  1. At Extreme: Price outside 2σ bands")
print(f"  2. Regime Valid: Theta >= {min_theta}, Half-Life <= {max_half_life}")
print(f"  3. Drift Dominant: (θ×|Xt-μ|)/σ > {z0}")
print(f"\nThis captures early reversal signals BEFORE price turns back.")
print("=" * 70)

all_signals = []
symbols_processed = 0
symbols_with_insufficient_data = []

for i, symbol in enumerate(symbols, 1):
    if i % 100 == 0:
        print(f"Processing {i}/{len(symbols)} symbols...")
    
    symbol_data = df[df['Symbol'] == symbol].copy()
    
    if len(symbol_data) < ma_length + reg_window + 1:
        symbols_with_insufficient_data.append(symbol)
        continue
    
    symbols_processed += 1
    signals = detect_signals(symbol_data)
    
    if signals:
        for signal in signals:
            signal_dict = {
                'Symbol': symbol,
                'Date': signal['Date'],
                'Close': signal['Close'],
                'Signal': signal['Signal'],
                'Theta': signal['Theta'],
                'Half_Life': signal['Half_Life'],
                'Deviation_Xt': signal['Deviation_Xt'],
                'Mu_OU': signal['Mu_OU'],
                'Sigma_OU': signal['Sigma_OU'],
                'Upper_Band': signal['Upper_Band'],
                'Lower_Band': signal['Lower_Band'],
                'Drift_Z': signal['Drift_Z'],
                'Regime_Valid': signal['Regime_Valid'],
                'Drift_Dominant': signal['Drift_Dominant']
            }
            if 'Filter_Reason' in signal:
                signal_dict['Filter_Reason'] = signal['Filter_Reason']
            all_signals.append(signal_dict)

print(f"\nProcessed {symbols_processed} symbols with sufficient data")

# Create results DataFrame
if all_signals:
    results_df = pd.DataFrame(all_signals)
    results_df = results_df.sort_values('Date', ascending=False)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    
    print(f"\n{'=' * 70}")
    print(f"SCAN COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    print(f"\nTotal Signals Found: {len(results_df)}")
    
    # Summary by signal type
    print("\nSignal Breakdown:")
    for signal_type in ['BUY', 'SELL', 'FILTERED_BUY', 'FILTERED_SELL']:
        count = len(results_df[results_df['Signal'] == signal_type])
        if count > 0:
            print(f"  {signal_type}: {count}")
    
    # Show active signals (BUY/SELL only)
    active_signals = results_df[results_df['Signal'].isin(['BUY', 'SELL'])]
    if len(active_signals) > 0:
        print(f"\n{'=' * 70}")
        print("ACTIVE SIGNALS (All 3 Factors Met):")
        print(f"{'=' * 70}")
        display_count = min(10, len(active_signals))
        for _, row in active_signals.head(display_count).iterrows():
            display_symbol = row['Symbol'].replace('.NS', '')
            print(f"\n{display_symbol} - {row['Signal']}")
            print(f"  Date: {row['Date'].strftime('%Y-%m-%d')}")
            print(f"  Close: ₹{row['Close']:.2f}")
            print(f"  ✓ Theta: {row['Theta']:.4f} (min: {min_theta})")
            print(f"  ✓ Half-Life: {row['Half_Life']:.2f} days (max: {max_half_life})")
            print(f"  ✓ Drift Z: {row['Drift_Z']:.2f} (threshold: {z0})")
            print(f"  Current Xt: {row['Deviation_Xt']:.4f}")
            print(f"  Distance from Mean: {abs(row['Deviation_Xt'] - row['Mu_OU']):.4f}")
            print(f"  Bands: [{row['Lower_Band']:.4f}, {row['Upper_Band']:.4f}]")
        
        if len(active_signals) > display_count:
            print(f"\n... and {len(active_signals) - display_count} more signals")
    else:
        print("\n" + "=" * 70)
        print("NO ACTIVE SIGNALS FOUND")
        print("=" * 70)
        
        # Analyze why signals are filtered
        filtered = results_df[results_df['Signal'].str.startswith('FILTERED_')]
        if len(filtered) > 0:
            print(f"\nFound {len(filtered)} filtered signals (at extremes but missing conditions).")
            if 'Filter_Reason' in filtered.columns:
                print("\nMost common filter reasons:")
                reasons = filtered['Filter_Reason'].value_counts()
                for reason, count in reasons.head(5).items():
                    print(f"  [{count}×] {reason}")
                
                # Analyze drift z-scores of filtered signals
                avg_drift_z = filtered['Drift_Z'].mean()
                print(f"\nFiltered signals average Drift Z-Score: {avg_drift_z:.2f}")
                print(f"Required threshold: {z0}")
                
                if avg_drift_z < z0:
                    print("→ Most signals filtered due to insufficient drift dominance")
                    print("  (mean-reversion force not strong enough vs noise)")
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 70}")
    
else:
    print("\n" + "=" * 70)
    print("NO SIGNALS FOUND")
    print("=" * 70)
    print("\nPossible reasons:")
    print("  • No stocks currently at extremes (outside 2σ bands)")
    print("  • Drift dominance threshold too strict (z-score > 1.28)")
    print("  • Regime filters excluding all candidates")

if symbols_with_insufficient_data:
    print(f"\n{len(symbols_with_insufficient_data)} symbols had insufficient data (need 261+ days)")

print("\nDone!")
