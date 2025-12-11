import os
import pandas as pd
from google import genai
from datetime import datetime
import time

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
OUTPUT_FILE = "outputs/research_report.md"
INPUT_FOLDER = "results_all"

# CORRECT STABLE MODEL
MODEL_PRIMARY = "gemini-2.5-pro"
# Fallback Model (Reliable & Fast)
MODEL_FALLBACK = "gemini-2.5-flash"

def get_buy_signals():
    """Finds today's CSV and extracts BUY signals"""
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(INPUT_FOLDER, f"ou_signals_all_{today}.csv")
    
    if not os.path.exists(csv_path):
        print(f"No signal file found for {today}")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        buys = df[df['Signal'] == 'BUY']
        if buys.empty:
            return []
        return [s.replace('.NS', '') for s in buys['Symbol'].unique()]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def run_research(symbols):
    """Sends signals to Gemini for Deep Analysis"""
    if not API_KEY:
        return "⚠️ Research skipped (No API Key)"

    # Initialize Client (New SDK)
    client = genai.Client(api_key=API_KEY)
    
    symbols_str = ", ".join(symbols)
    
    prompt = f"""
    Role: Act as a Senior Equity Research Analyst for a top-tier Indian institutional fund. 
    Task: Generate a "Strategic Equity Research Report" for: {symbols_str}.

    Tone: Professional, ruthless, analytical. Use terms like "Capital Efficiency," "Moat Durability," "Sovereign Moat."

    Report Structure (Use Markdown formatting heavily):
    
    # Executive Summary
    Define the market phase and dominant themes (e.g., Infra Super-Cycle, Rural Distress).
    
    # Thematic Landscape
    Group stocks into themes. Identify Leaders vs Laggards.

    # Fundamental Data Matrix
    (Create a Markdown Table with columns: Ticker, P/E, ROCE %, Debt/Equity, Pledge %, Moat Rating, Verdict).

    # High Conviction Ideas (Top 3)
    Analyze Cash Flow (OCF/EBITDA), Moat, and Valuation.

    # Red Flags & Avoids
    Highlight Pledge Traps (>10%) or Capital Destroyers.

    # Investment Verdicts
    * **Tier 1 (Strategic Buys):** ...
    * **Tier 2 (Accumulate):** ...
    * **Tier 3 (Avoid):** ...

    Constraints:
    - Use latest data.
    - Be decisive (Buy/Sell/Avoid).
    """

    # Try Primary Model (Gemini 2.5 Pro)
    try:
        print(f"Running research on: {symbols_str}")
        print(f"Attempting with {MODEL_PRIMARY}...")
        
        response = client.models.generate_content(
            model=MODEL_PRIMARY,
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        print(f"⚠️ Primary model ({MODEL_PRIMARY}) failed: {e}")
        print(f"🔄 Switching to Fallback: {MODEL_FALLBACK}...")
        
        try:
            response = client.models.generate_content(
                model=MODEL_FALLBACK,
                contents=prompt
            )
            return response.text
        except Exception as e2:
            return f"⚠️ Research Failed completely: {e2}"

def main():
    os.makedirs("outputs", exist_ok=True)
    signals = get_buy_signals()
    
    if not signals:
        print("No BUY signals to research.")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("NO_SIGNALS")
        return

    print(f"Found {len(signals)} BUY signals. Starting Analyst Agent...")
    report = run_research(signals)
    
    header = f"""% Strategic Equity Research Report
% AI Research Desk
% {datetime.now().strftime('%Y-%m-%d')}

"""
    full_report = header + report
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    
    print(f"Research complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
