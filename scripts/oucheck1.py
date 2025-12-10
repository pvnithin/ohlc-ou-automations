import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
# Changed base folder to Option A data folder
base_folder = os.path.join(os.getcwd(), "data")
input_parquet_name = "daily_ohlc.parquet"
# Write results to results_selected (Option A)
results_folder = "results_selected"  # Folder to store results

# OU Process Parameters (matching Pine Script)
reg_window = 60  # Estimation Lookback Window
ma_length = 200  # Long-term Mean Length
min_theta = 0.02  # Min Theta for Signal
max_half_life = 25  # Max Half-Life for Signal

# Symbols to scan (will add .NS suffix automatically)
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

# --- End Configuration ---

input_path = os.path.join(base_folder, input_parquet_name)

# Create results folder if it doesn't exist
results_path = os.path.join(os.getcwd(), results_folder)
os.makedirs(results_path, exist_ok=True)

# Generate filename with current date
current_date = datetime.now().strftime('%Y-%m-%d')
output_filename = f"ou_signals_{current_date}.csv"
output_path = os.path.join(results_path, output_filename)

def calculate_ou_parameters(prices, reg_window, ma_length):
    """
    Calculate OU process parameters for a price series
    Returns: theta, mu_ou, sigma_ou, half_life, Xt, band_upper, band_lower
    """
    if len(prices) < ma_length + reg_window + 1:
        return None
    
    # Calculate long-term mean
    long_term_mean = prices.rolling(window=ma_length).mean()
    
    # Deviation from mean
    Xt = prices - long_term_mean
    
    # Change in deviation
    dX = Xt.diff()
    X_prev = Xt.shift(1)
    
    # Get the last reg_window points for regression
    last_idx = len(prices) - 1
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
    
    # Get current values
    current_Xt = Xt.iloc[-1]
    prev_Xt = Xt.iloc[-2] if len(Xt) > 1 else None
    
    return {
        'theta': theta,
        'mu_ou': mu_ou,
        'sigma_ou': sigma_ou,
        'half_life': half_life,
        'Xt': current_Xt,
        'prev_Xt': prev_Xt,
        'band_upper': band_upper,
        'band_lower': band_lower
    }

def detect_signals(symbol_data):
    """
    Detect buy/sell signals for a single symbol
    """
    results = []
    
    # Sort by date
    symbol_data = symbol_data.sort_values('Date').reset_index(drop=True)
    
    if len(symbol_data) < ma_length + reg_window + 1:
        return results
    
    # Calculate OU parameters
    ou_params = calculate_ou_parameters(
        symbol_data['Close'],
        reg_window,
        ma_length
    )
    
    if ou_params is None:
        return results
    
    theta = ou_params['theta']
    half_life = ou_params['half_life']
    Xt = ou_params['Xt']
    prev_Xt = ou_params['prev_Xt']
    band_upper = ou_params['band_upper']
    band_lower = ou_params['band_lower']
    
    # Check regime validity
    regime_valid = (theta >= min_theta) and (half_life <= max_half_life)
    
    # Detect crossovers
    raw_buy = False
    raw_sell = False
    
    if prev_Xt is not None and not np.isnan(prev_Xt):
        # Crossunder: previous was above, current is below
        raw_buy = (prev_Xt >= band_lower) and (Xt < band_lower)
        # Crossover: previous was below, current is above
        raw_sell = (prev_Xt <= band_upper) and (Xt > band_upper)
    
    # Final signals
    buy_signal = raw_buy and regime_valid
    sell_signal = raw_sell and regime_valid
    
    if buy_signal or sell_signal:
        signal_type = 'BUY' if buy_signal else 'SELL'
        results.append({
            'Signal': signal_type,
            'Theta': theta,
            'Half_Life': half_life,
            'Deviation_Xt': Xt,
            'Upper_Band': band_upper,
            'Lower_Band': band_lower,
            'Regime_Valid': regime_valid
        })
    elif raw_buy or raw_sell:
        # Filtered signal (regime not valid)
        signal_type = 'FILTERED_BUY' if raw_buy else 'FILTERED_SELL'
        results.append({
            'Signal': signal_type,
            'Theta': theta,
            'Half_Life': half_life,
            'Deviation_Xt': Xt,
            'Upper_Band': band_upper,
            'Lower_Band': band_lower,
            'Regime_Valid': regime_valid
        })
    
    return results

# --- Main Execution ---
print(f"OU Process Signal Scanner")
print(f"Scan Date: {current_date}")
print(f"Results will be saved to: {results_folder}/{output_filename}")
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

# Filter for only the specified symbols
# The symbols in the parquet file should already have .NS suffix from yfinance
available_symbols = set(df['Symbol'].unique())

# Create both versions (with and without .NS) to match against parquet
symbols_to_check = set()
for sym in SYMBOLS_TO_SCAN:
    symbols_to_check.add(sym)
    symbols_to_check.add(sym + '.NS')

# Find which symbols are available
symbols = list(available_symbols & symbols_to_check)

if not symbols:
    print("ERROR: None of the specified symbols found in the parquet file!")
    print(f"Available symbols in file: {sorted(list(available_symbols))[:10]}...")
    exit()

symbols_not_found = [s for s in SYMBOLS_TO_SCAN if s not in available_symbols and s + '.NS' not in available_symbols]
if symbols_not_found:
    print(f"\nWarning: {len(symbols_not_found)} symbols not found in parquet file:")
    print(", ".join(symbols_not_found[:20]))
    if len(symbols_not_found) > 20:
        print(f"... and {len(symbols_not_found) - 20} more")

print(f"\nScanning {len(symbols)} symbols for OU signals...")
print(f"Parameters: Theta >= {min_theta}, Half-Life <= {max_half_life}")
print("=" * 70)

all_signals = []
symbols_processed = 0
symbols_with_insufficient_data = []

for i, symbol in enumerate(symbols, 1):
    if i % 50 == 0:
        print(f"Processing {i}/{len(symbols)} symbols...")
    
    symbol_data = df[df['Symbol'] == symbol].copy()
    
    if len(symbol_data) < ma_length + reg_window + 1:
        symbols_with_insufficient_data.append(symbol)
        continue
    
    symbols_processed += 1
    signals = detect_signals(symbol_data)
    
    if signals:
        for signal in signals:
            latest_date = symbol_data['Date'].max()
            latest_close = symbol_data['Close'].iloc[-1]
            
            all_signals.append({
                'Symbol': symbol,
                'Date': latest_date,
                'Close': latest_close,
                'Signal': signal['Signal'],
                'Theta': signal['Theta'],
                'Half_Life': signal['Half_Life'],
                'Deviation_Xt': signal['Deviation_Xt'],
                'Upper_Band': signal['Upper_Band'],
                'Lower_Band': signal['Lower_Band'],
                'Regime_Valid': signal['Regime_Valid']
            })

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
        print("ACTIVE SIGNALS (Valid Regime):")
        print(f"{'=' * 70}")
        for _, row in active_signals.iterrows():
            # Remove .NS suffix for cleaner display
            display_symbol = row['Symbol'].replace('.NS', '')
            print(f"\n{display_symbol} - {row['Signal']}")
            print(f"  Date: {row['Date'].strftime('%Y-%m-%d')}")
            print(f"  Close: {row['Close']:.2f}")
            print(f"  Theta: {row['Theta']:.4f}")
            print(f"  Half-Life: {row['Half_Life']:.2f}")
            print(f"  Deviation: {row['Deviation_Xt']:.4f}")
            print(f"  Bands: [{row['Lower_Band']:.4f}, {row['Upper_Band']:.4f}]")
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_path}")
    
else:
    print("\nNo signals found in any symbols.")
    print("This could mean:")
    print("  - No symbols are currently crossing the bands")
    print("  - All signals are filtered due to regime criteria")

if symbols_with_insufficient_data:
    print(f"\n{len(symbols_with_insufficient_data)} symbols had insufficient data (need 261+ days)")

print("\nDone!")
