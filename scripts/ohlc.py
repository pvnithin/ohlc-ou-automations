import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

# --- Configuration ---
lookback_period = "5y" 
# Use relative paths from scripts folder
script_dir = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_dir)  # Parent directory (repo root)
data_folder = os.path.join(base_folder, "data")
input_csv_name = "nse stocks.csv"
output_parquet_name = "daily_ohlc.parquet"
failure_log_name = "failed_symbols.log"

# Add retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
# --- End Configuration ---

# Construct full paths
csv_path = os.path.join(data_folder, input_csv_name)
output_path = os.path.join(data_folder, output_parquet_name)
log_path = os.path.join(data_folder, failure_log_name)

# Create data folder if it doesn't exist
os.makedirs(data_folder, exist_ok=True)

print(f"{'='*70}")
print(f"STOCK DATA FETCHER - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")

# --- 1. Read Master List of Symbols ---
try:
    df = pd.read_csv(csv_path)
    all_symbols_master_list = set(df['SYMBOL'].dropna().unique())
    print(f"✓ Loaded {len(all_symbols_master_list)} symbols from CSV")
except FileNotFoundError:
    print(f"✗ Error: Input file not found at {csv_path}")
    print(f"Please make sure '{input_csv_name}' is in the 'data' folder.")
    exit(1)
except Exception as e:
    print(f"✗ Error reading CSV: {e}")
    exit(1)

# --- 2. Check for Existing Data ---
existing_df = pd.DataFrame()
symbols_to_fetch_full = []  # Symbols not in parquet at all
symbols_to_update = {}  # {symbol: last_date} for incremental updates

if os.path.exists(output_path):
    try:
        print(f"\nLoading existing data from {output_parquet_name}...")
        existing_df = pd.read_parquet(output_path)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # Get the last date for each symbol
        last_dates = existing_df.groupby('Symbol')['Date'].max()
        existing_symbols = set(last_dates.index)
        
        print(f"✓ Found {len(existing_symbols)} symbols in existing file")
        print(f"  Latest date in file: {existing_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Symbols not in the file at all
        symbols_to_fetch_full = list(all_symbols_master_list - existing_symbols)
        
        # Symbols that need updating (fetch from last date to today)
        today = datetime.now().date()
        for sym in all_symbols_master_list & existing_symbols:
            last_date = last_dates[sym]
            # Only update if data is more than 1 day old
            if last_date.date() < today - timedelta(days=1):
                symbols_to_update[sym] = last_date
        
        print(f"✓ New symbols to fetch: {len(symbols_to_fetch_full)}")
        print(f"✓ Existing symbols to update: {len(symbols_to_update)}")
        
    except Exception as e:
        print(f"⚠️ Error reading existing Parquet file: {e}")
        print("Will attempt to fetch ALL symbols from CSV.")
        symbols_to_fetch_full = list(all_symbols_master_list)
else:
    print("\nNo existing data file found. Will fetch all symbols.")
    symbols_to_fetch_full = list(all_symbols_master_list)

def fetch_with_retry(ticker_obj, **kwargs):
    """Fetch data with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            hist = ticker_obj.history(**kwargs)
            if not hist.empty:
                return hist
            print(f"    Attempt {attempt + 1}: Empty data returned")
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    
    return pd.DataFrame()  # Return empty DataFrame after all retries

# --- 3. Fetch Missing Data ---
new_data_list = []
failed_symbols_log = []

# 3a. Fetch FULL data for new symbols
if symbols_to_fetch_full:
    print(f"\n{'='*70}")
    print(f"FETCHING FULL DATA FOR {len(symbols_to_fetch_full)} NEW SYMBOLS")
    print(f"Lookback period: {lookback_period}")
    print(f"{'='*70}")

    for idx, sym in enumerate(symbols_to_fetch_full, 1):
        try:
            symbol_yf = sym if sym.endswith('.NS') else sym + ".NS"
            print(f"[{idx}/{len(symbols_to_fetch_full)}] Fetching {sym}...", end=" ")
            
            ticker = yf.Ticker(symbol_yf)
            hist = fetch_with_retry(ticker, period=lookback_period, interval="1d")
            
            if hist.empty:
                print(f"✗ No data")
                failed_symbols_log.append(sym)
                continue

            hist = hist.reset_index()
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            
            hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            hist['Symbol'] = sym
            
            new_data_list.append(hist)
            print(f"✓ {len(hist)} days")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_symbols_log.append(f"{sym} (Error: {e})")

# 3b. Fetch INCREMENTAL data for existing symbols
if symbols_to_update:
    print(f"\n{'='*70}")
    print(f"UPDATING {len(symbols_to_update)} EXISTING SYMBOLS")
    print(f"{'='*70}")
    
    for idx, (sym, last_date) in enumerate(symbols_to_update.items(), 1):
        try:
            symbol_yf = sym if sym.endswith('.NS') else sym + ".NS"
            
            # Calculate start date (day after last_date)
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            print(f"[{idx}/{len(symbols_to_update)}] Updating {sym} from {start_date}...", end=" ")
            
            ticker = yf.Ticker(symbol_yf)
            hist = fetch_with_retry(ticker, start=start_date, end=end_date, interval="1d")
            
            if hist.empty:
                print(f"✓ Already up-to-date")
                continue

            hist = hist.reset_index()
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            
            hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            hist['Symbol'] = sym
            
            new_data_list.append(hist)
            print(f"✓ Added {len(hist)} new days")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_symbols_log.append(f"{sym} (Update Error: {e})")

# --- 4. Save Failure Log ---
if failed_symbols_log:
    print(f"\n⚠️ Writing {len(failed_symbols_log)} failed symbols to {failure_log_name}")
    try:
        with open(log_path, 'w') as f:
            for item in failed_symbols_log:
                f.write(f"{item}\n")
    except Exception as e:
        print(f"✗ Error writing log file: {e}")
else:
    print(f"\n✓ No failures to log")

# --- 5. Combine and Save Final Data ---
if new_data_list:
    print(f"\n{'='*70}")
    print("COMBINING AND SAVING DATA")
    print(f"{'='*70}")
    
    new_df = pd.concat(new_data_list, ignore_index=True)
    print(f"✓ Fetched {len(new_df)} new records")
    
    # Combine old and new
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    before_dedup = len(final_df)
    final_df = final_df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')
    after_dedup = len(final_df)
    
    if before_dedup > after_dedup:
        print(f"✓ Removed {before_dedup - after_dedup} duplicate records")
    
    # Sort by Symbol and then Date
    final_df = final_df.sort_values(by=['Symbol', 'Date'])
    
    print(f"\n{'='*70}")
    print("FINAL DATABASE SUMMARY")
    print(f"{'='*70}")
    print(f"Total records: {len(final_df):,}")
    print(f"Total symbols: {final_df['Symbol'].nunique()}")
    print(f"Date range: {final_df['Date'].min().strftime('%Y-%m-%d')} to {final_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Latest data date: {final_df['Date'].max().strftime('%Y-%m-%d')}")
    
    try:
        final_df.to_parquet(output_path, index=False, engine='pyarrow')
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"\n✓ Successfully saved to {output_path}")
        print(f"  File size: {file_size:.2f} MB")
    except ImportError:
        print("\n✗ Error: 'pyarrow' library not found.")
        print("Please install it: pip install pyarrow")
        exit(1)
    except Exception as e:
        print(f"\n✗ Error saving Parquet file: {e}")
        exit(1)
else:
    print(f"\n{'='*70}")
    print("NO NEW DATA FETCHED")
    print(f"{'='*70}")
    if not existing_df.empty:
        print("The existing file remains unchanged.")
        print(f"Latest data date: {existing_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Check if data is stale
        today = datetime.now().date()
        latest_date = existing_df['Date'].max().date()
        days_old = (today - latest_date).days
        
        if days_old > 1:
            print(f"\n⚠️ WARNING: Data is {days_old} days old!")
            print("This may cause incorrect signals. Market may be closed or fetch failed.")
        else:
            print("\n✓ All data is up to date!")

print(f"\n{'='*70}")
print(f"COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
