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

# INPUT: Look in results_all (Output of oucheck.py)
INPUT_FOLDER = os.path.join(BASE_DIR, "results_all")
# OUTPUT: Save Excel to results_all
OUTPUT_FOLDER = os.path.join(BASE_DIR, "results_all")

def get_latest_signal_file():
    """Finds the latest ou_signals_all_*.csv file."""
    search_pattern = os.path.join(INPUT_FOLDER, "ou_signals_all_*.csv")
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
        print("No 'ou_signals_all_*.csv' found in results_all/")
        sys.exit(0) # Exit cleanly

    print(f"Processing file: {os.path.basename(input_csv)}")
    df = pd.read_csv(input_csv)

    # 2. Filter for BUY signals
    if 'Signal' not in df.columns or 'Symbol' not in df.columns:
        print("CSV missing 'Signal' or 'Symbol' columns.")
        sys.exit(1)

    buy_df = df[df['Signal'] == 'BUY']
    symbols = buy_df['Symbol'].unique()

    if len(symbols) == 0:
        print("No BUY signals found to scrape.")
        sys.exit(0)

    # 3. Setup Output File
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_filename = f"Screener_Analysis_All_{today_str}.xlsx"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    print(f"Found {len(symbols)} companies for detailed analysis.")

    # 4. Scrape and Write to Excel
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # -- Summary Sheet --
        summary_cols = [c for c in ['Symbol', 'Date', 'Close', 'Deviation_Xt', 'Lower_Band'] if c in buy_df.columns]
        summary_df = buy_df[summary_cols].copy()
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # -- Individual Company Sheets --
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] Scraping {symbol}...")
            
            company_data = get_company_data(symbol)
            if not company_data:
                continue
                
            sheet_name = company_data['symbol'][:30] # Excel sheet name limit
            row = 0
            
            # Header
            header_data = [
                ["Symbol", company_data['symbol']],
                ["Sector", company_data['breadcrumbs']],
                ["About", company_data['about']]
            ]
            pd.DataFrame(header_data).to_excel(writer, sheet_name=sheet_name, startrow=row, index=False, header=False)
            row += 4

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

    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    main()