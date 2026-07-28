"""main.py
Entry point for the Options Scanner. Scans a hardcoded watchlist and prints a terminal report of option contracts
that show a positive volatility edge (True Expected Vol - Market IV >= 4 percentage points).

Usage:
  python main.py

This script uses only free, zero-registration data via yfinance.
"""

from scanner_engine.data_fetcher import fetch_historical_prices, fetch_options_chain
from scanner_engine.math_engine import GARCHEngine

import datetime

WATCHLIST = ["AAPL", "NVDA", "AMD", "TSLA", "MSFT", "META", "AMZN"]
EDGE_THRESHOLD = 0.04  # 4% in decimal

def format_pct(x):
    try:
        return f"{float(x)*100:5.2f}%"
    except Exception:
        return "-"

def print_report(recs):
    title = "🔥 TOP QUANT OPTIONS RECOMMENDATIONS FOR MANUALLY BUYING RIGHT NOW 🔥"
    print("\n" + title)
    print("=" * len(title))

    if not recs:
        print("No recommendations found that meet the edge threshold. Try running again later.")
        return

    # Column widths
    cols = ["Ticker", "Type", "Strike", "Expiration", "Market IV", "Predicted Vol", "Edge %"]
    widths = [8, 6, 10, 12, 12, 14, 10]

    # Header
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    for r in recs:
        line = (
            f"{r['ticker']:<8} | {r['option_type']:<6} | {r['strike']:<10.2f} | {r['expiration']:<12} | "
            f"{format_pct(r['market_iv']):<12} | {format_pct(r['pred_vol']):<14} | {r['edge']*100:6.2f}%"
        )
        print(line)

def main():
    recommendations = []
    today = datetime.datetime.utcnow().date()

    for ticker in WATCHLIST:
        print(f"Scanning {ticker}...")
        try:
            prices = fetch_historical_prices(ticker)
            if prices is None or len(prices) < 100:
                print(f"  Skipping {ticker}: not enough historical data.")
                continue

            options_df = fetch_options_chain(ticker)
            if options_df is None or options_df.empty:
                print(f"  Skipping {ticker}: no options data available.")
                continue

            g = GARCHEngine(prices)
            pred_vol = g.forecast_next_day_annual_vol()
            if pred_vol is None:
                print(f"  Skipping {ticker}: GARCH forecast failed.")
                continue

            # For each expiration within desired DTE, find ATM strike and evaluate both call & put
            expirations = options_df['expiration'].dropna().unique()
            for exp in expirations:
                try:
                    exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
                except Exception:
                    continue

                dte = (exp_date - today).days
                if dte < 35 or dte > 55:
                    continue

                # Filter this expiration
                df_e = options_df[options_df['expiration'] == exp].copy()
                if df_e.empty:
                    continue

                # Current price: use last close from prices series
                current_price = float(prices.iloc[-1])

                # Find ATM strike (closest)
                df_e['strike_diff'] = (df_e['strike'] - current_price).abs()
                min_diff = df_e['strike_diff'].min()
                atm_df = df_e[df_e['strike_diff'] == min_diff]

                # Iterate ATM rows (calls and puts)
                for _, row in atm_df.iterrows():
                    option_type = row.get('contractType') or row.get('type') or row.get('side') or row.get('option_type') or 'CALL'
                    market_iv = row.get('impliedVolatility')
                    if market_iv is None or market_iv != market_iv:  # NaN check
                        continue

                    edge = pred_vol - float(market_iv)
                    if edge >= EDGE_THRESHOLD:
                        recommendations.append({
                            'ticker': ticker,
                            'option_type': 'Call' if str(option_type).lower().startswith('c') else 'Put',
                            'strike': float(row['strike']),
                            'expiration': exp_date.isoformat(),
                            'market_iv': float(market_iv),
                            'pred_vol': float(pred_vol),
                            'edge': float(edge),
                        })

        except Exception as e:
            print(f"  Error scanning {ticker}: {e}")

    # Sort recommendations by edge descending
    recommendations = sorted(recommendations, key=lambda x: x['edge'], reverse=True)
    print_report(recommendations)

if __name__ == '__main__':
    main()
