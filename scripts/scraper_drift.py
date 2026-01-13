import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
from io import StringIO
import os
import glob
from datetime import datetime
import sys

# --- Configuration ---
# Use relative paths to ensure it works in GitHub Actions
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# INPUT: Look in results_drift_dominance (Output of drift dominance scanner)
INPUT_FOLDER = os.path.join(BASE_DIR, "results_drift_dominance")
# OUTPUT: Save Excel to results_drift_dominance
OUTPUT_FOLDER = os.path.join(BASE_DIR, "results_drift_dominance")

def get_latest_signal_file():
    """Finds the latest ou_signals_drift_*.csv file."""
    search_pattern = os.path.join(INPUT_FOLDER, "ou_signals_drift_*.csv")
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def get_company_data(symbol):
    """Scrapes comprehensive data for a single company."""
    # Clean symbol (remove .NS/.BO)
    clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
    url = f"https://www.screener.in/company/{clean_symbol}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data = {
            'symbol': clean_symbol,
            'breadcrumbs': '',
            'about': '',
            'ratios': {},
            'pros': [],
            'cons': [],
            'tables': []
        }

        # 1. Breadcrumbs (Sector/Industry)
        breadcrumbs_div = soup.find('div', {'class': 'breadcrumbs'})
        if breadcrumbs_div:
            data['breadcrumbs'] = " > ".join([a.get_text(strip=True) for a in breadcrumbs_div.find_all('a')])

        # 2. About
        about_div = soup.find('div', {'class': 'company-profile-about'})
        if about_div:
            data['about'] = about_div.get_text(strip=True)

        # 3. Ratios
        ratios_div = soup.find('div', {'class': 'company-ratios'})
        if ratios_div:
            ul = ratios_div.find('ul', {'id': 'top-ratios'})
            if ul:
                for li in ul.find_all('li'):
                    name = li.find('span', {'class': 'name'})
                    val = li.find('span', {'class': 'value'})
                    if name and val:
                        data['ratios'][name.get_text(strip=True)] = val.get_text(strip=True)

        # 4. Pros & Cons
        pros_div = soup.find('div', {'class': 'pros'})
        if pros_div:
            data['pros'] = [li.get_text(strip=True) for li in pros_div.find_all('li')]
        cons_div = soup.find('div', {'class': 'cons'})
        if cons_div:
            data['cons'] = [li.get_text(strip=True) for li in cons_div.find_all('li')]

        # 5. Tables
        try:
            dfs = pd.read_html(StringIO(response.text))
            data['tables'] = dfs
        except ValueError:
            pass

        return data

    except Exception as e:
        print(f"Error scraping {clean_symbol}: {e}")
        return None

def main():
    # 1. Identify Input File
    input_csv = get_latest_signal_file()
    if not input_csv:
        print("No 'ou_signals_drift_*.csv' found in results_drift_dominance/")
        sys.exit(0) # Exit cleanly

    print(f"Processing file: {os.path.basename(input_csv)}")
    df = pd.read_csv(input_csv)

    # 2. Filter for BUY signals only (all 3 conditions met)
    if 'Signal' not in df.columns or 'Symbol' not in df.columns:
        print("CSV missing 'Signal' or 'Symbol' columns.")
        sys.exit(1)

    # Only process BUY signals (not FILTERED_BUY)
    buy_df = df[df['Signal'] == 'BUY'].copy()
    symbols = buy_df['Symbol'].unique()

    if len(symbols) == 0:
        print("No BUY signals found to scrape.")
        print("(All signals may be filtered - check ou_signals_drift_*.csv for FILTERED_BUY entries)")
        sys.exit(0)

    # 3. Setup Output File
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_filename = f"Screener_Analysis_Drift_{today_str}.xlsx"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    print(f"Found {len(symbols)} companies with valid drift dominance signals.")
    print(f"(All meet: At Extreme + Regime Valid + Drift Dominant)")

    # 4. Scrape and Write to Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # -- Summary Sheet with Drift Dominance Metrics --
        summary_cols = [
            'Symbol', 'Date', 'Close', 'Signal',
            'Theta', 'Half_Life', 'Drift_Z',
            'Deviation_Xt', 'Lower_Band', 'Upper_Band',
            'Regime_Valid', 'Drift_Dominant'
        ]
        # Include only columns that exist in the DataFrame
        summary_cols = [c for c in summary_cols if c in buy_df.columns]
        summary_df = buy_df[summary_cols].copy()
        
        # Add a helper column showing distance from mean
        if 'Deviation_Xt' in summary_df.columns and 'Mu_OU' in buy_df.columns:
            summary_df['Distance_From_Mean'] = abs(buy_df['Deviation_Xt'] - buy_df['Mu_OU'])
        
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the summary sheet
        workbook = writer.book
        worksheet = writer.sheets['Summary']
        
        # Add a header format
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        # Auto-adjust column widths
        for idx, col in enumerate(summary_df.columns):
            max_len = max(
                summary_df[col].astype(str).apply(len).max(),
                len(col)
            ) + 2
            worksheet.set_column(idx, idx, max_len)

        # -- Individual Company Sheets --
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] Scraping {symbol}...")
            
            # Get signal data for this symbol
            signal_row = buy_df[buy_df['Symbol'] == symbol].iloc[0]
            
            company_data = get_company_data(symbol)
            if not company_data:
                continue
                
            sheet_name = company_data['symbol'][:30] # Excel sheet name limit
            row = 0
            
            # Header with Signal Information
            header_data = [
                ["Symbol", company_data['symbol']],
                ["Sector", company_data['breadcrumbs']],
                ["Signal Date", signal_row['Date']],
                ["Close Price", f"₹{signal_row['Close']:.2f}"],
                [""],
                ["OU Process Metrics"],
                ["Theta", f"{signal_row['Theta']:.4f}"],
                ["Half-Life", f"{signal_row['Half_Life']:.2f} days"],
                ["Drift Z-Score", f"{signal_row['Drift_Z']:.2f} (threshold: 1.28)"],
                ["Deviation (Xt)", f"{signal_row['Deviation_Xt']:.4f}"],
                ["Lower Band", f"{signal_row['Lower_Band']:.4f}"],
                [""],
                ["About", company_data['about']]
            ]
            pd.DataFrame(header_data).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
            row += len(header_data) + 1

            # Ratios
            if company_data['ratios']:
                pd.DataFrame(["Key Ratios"]).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
                row += 1
                ratios_df = pd.DataFrame(list(company_data['ratios'].items()), columns=['Ratio', 'Value'])
                ratios_df.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
                row += len(ratios_df) + 2

            # Pros/Cons
            if company_data['pros'] or company_data['cons']:
                pd.DataFrame(["Analysis"]).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
                row += 1
                if company_data['pros']:
                    pd.DataFrame([["Pros"]] + [[p] for p in company_data['pros']]).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
                    row += len(company_data['pros']) + 1
                if company_data['cons']:
                    pd.DataFrame([["Cons"]] + [[c] for c in company_data['cons']]).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
                    row += len(company_data['cons']) + 2

            # Financial Tables
            for idx, tbl in enumerate(company_data['tables']):
                title = f"Table {idx+1}"
                pd.DataFrame([title]).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
                row += 1
                tbl.to_excel(writer, sheet_name=sheet_name, startrow=row, index=False)
                row += len(tbl) + 2
            
            # Respect rate limits
            time.sleep(random.uniform(1.5, 3))

    print(f"\n{'='*70}")
    print(f"Analysis Complete!")
    print(f"{'='*70}")
    print(f"Processed {len(symbols)} companies with drift dominance signals")
    print(f"Excel file saved to: {output_path}")
    print(f"\nEach company met all 3 criteria:")
    print(f"  ✓ Price at extreme (outside 2σ bands)")
    print(f"  ✓ Strong mean reversion (theta ≥ 0.02, half-life ≤ 25)")
    print(f"  ✓ Drift dominant (z-score > 1.28)")

if __name__ == "__main__":
    main()
