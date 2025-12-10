import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Configuration ---
lookback_period = "5y" 
# Changed to Option A: store data under repository ./data
base_folder = os.path.join(os.getcwd(), "data")
os.makedirs(base_folder, exist_ok=True)

input_csv_name = "nse stocks.csv"
output_parquet_name = "daily_ohlc.parquet"
failure_log_name = "failed_symbols.log"
# --- End Configuration ---

# Construct full paths
csv_path = os.path.join(base_folder, input_csv_name)
output_path = os.path.join(base_folder, output_parquet_name)
log_path = os.path.join(base_folder, failure_log_name)

# --- 1. Read Master List of Symbols ---
try:
    df = pd.read_csv(csv_path)
    all_symbols_master_list = set(df['SYMBOL'].dropna().unique())
except FileNotFoundError:
    print(f"Error: Input file not found at {csv_path}")
    print(f"Please make sure '{input_csv_name}' is in the '{base_folder}' folder.")
    exit()

# --- 2. Check for Existing Data ---
existing_df = pd.DataFrame()
symbols_to_fetch_full = []  # Symbols not in parquet at all
symbols_to_update = {}  # {symbol: last_date} for incremental updates

if os.path.exists(output_path):
    try:
        print(f"Loading existing data from {output_parquet_name}...")
        existing_df = pd.read_parquet(output_path)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
        
        # Get the last date for each symbol
        last_dates = existing_df.groupby('Symbol')['Date'].max()
        existing_symbols = set(last_dates.index)
        
        print(f"Found {len(existing_symbols)} symbols in existing file.")
        
        # Symbols not in the file at all
        symbols_to_fetch_full = list(all_symbols_master_list - existing_symbols)
        
        # Symbols that need updating (fetch from last date to today)
        for sym in all_symbols_master_list & existing_symbols:
            last_date = last_dates[sym]
            symbols_to_update[sym] = last_date
        
        print(f"New symbols to fetch: {len(symbols_to_fetch_full)}")
        print(f"Existing symbols to update: {len(symbols_to_update)}")
        
    except Exception as e:
        print(f"Error reading existing Parquet file {output_path}: {e}")
        print("Will attempt to fetch ALL symbols from CSV.")
        symbols_to_fetch_full = list(all_symbols_master_list)
else:
    print("No existing data file found. Will attempt to fetch all symbols.")
    symbols_to_fetch_full = list(all_symbols_master_list)

# --- 3. Fetch Missing Data ---
new_data_list = []
failed_symbols_log = []

# 3a. Fetch FULL data for new symbols
if symbols_to_fetch_full:
    print(f"\n=== Fetching FULL data for {len(symbols_to_fetch_full)} new symbols ===")
    print(f"Lookback period: {lookback_period}")

    for sym in symbols_to_fetch_full:
        try:
            symbol_yf = sym if sym.endswith('.NS') else sym + ".NS"
            print(f"Fetching {sym} (full history)...")
            
            ticker = yf.Ticker(symbol_yf)
            hist = ticker.history(period=lookback_period, interval="1d")
            
            if hist.empty:
                print(f"  No data for {sym}")
                failed_symbols_log.append(sym)
                continue

            hist = hist.reset_index()
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            
            hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            hist['Symbol'] = sym
            
            new_data_list.append(hist)
            print(f"  ✓ Fetched {len(hist)} days")
            
        except Exception as e:
            print(f"  ✗ Error fetching {sym}: {e}")
            failed_symbols_log.append(f"{sym} (Error: {e})")

# 3b. Fetch INCREMENTAL data for existing symbols
if symbols_to_update:
    print(f"\n=== Updating {len(symbols_to_update)} existing symbols with recent data ===")
    
    for sym, last_date in symbols_to_update.items():
        try:
            symbol_yf = sym if sym.endswith('.NS') else sym + ".NS"
            
            # Calculate start date (day after last_date)
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            print(f"Updating {sym} from {start_date} to {end_date}...")
            
            ticker = yf.Ticker(symbol_yf)
            hist = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if hist.empty:
                print(f"  No new data for {sym}")
                continue

            hist = hist.reset_index()
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].dt.tz_localize(None)
            
            hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            hist['Symbol'] = sym
            
            new_data_list.append(hist)
            print(f"  ✓ Added {len(hist)} new days")
            
        except Exception as e:
            print(f"  ✗ Error updating {sym}: {e}")
            failed_symbols_log.append(f"{sym} (Update Error: {e})")

# --- 4. Save Failure Log ---
if failed_symbols_log:
    print(f"\nWriting {len(failed_symbols_log)} failed symbols to {log_path}")
    try:
        with open(log_path, 'w') as f:
            for item in failed_symbols_log:
                f.write(f"{item}\n")
    except Exception as e:
        print(f"Error writing log file: {e}")
else:
    print("\nNo failures to log.")

# --- 5. Combine and Save Final Data ---
if new_data_list:
    print("\n=== Combining new data with existing data ===")
    new_df = pd.concat(new_data_list, ignore_index=True)
    
    # Combine old and new
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (in case of overlapping dates)
    final_df = final_df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')
    
    # Sort by Symbol and then Date
    final_df = final_df.sort_values(by=['Symbol', 'Date'])
    
    print(f"Total records: {len(final_df)}")
    print(f"Total symbols: {final_df['Symbol'].nunique()}")
    
    try:
        final_df.to_parquet(output_path, index=False, engine='pyarrow')
        print(f"\n✓ Done. All data saved to {output_path}")
    except ImportError:
        print("\nError: 'pyarrow' library not found.")
        print("Please install it to save as Parquet: pip install pyarrow")
    except Exception as e:
        print(f"\nError saving Parquet file: {e}")
else:
    print("\nNo new data was fetched.")
    if not existing_df.empty:
        print("The existing file remains unchanged.")
        print("All data is up to date!")
