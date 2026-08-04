"""main.py
Updated entry point: scan the S&P 500 (or tickers.txt) and evaluate all option contracts priced under $1000.

Changes made:
- If tickers.txt exists, use it. Otherwise fetch S&P 500 list from Wikipedia.
- Removed the strict ATM + DTE (35-55) restriction: evaluate all expirations and all strikes.
- For each option, compute an option "price" using mid=(bid+ask)/2 when available, or lastPrice as a fallback.
  Only keep contracts with price < 1000.
- Use a bounded ThreadPoolExecutor to scan tickers in parallel (small concurrency) and a short sleep
  between tickers to avoid hammering yfinance.
- Still uses the GARCHEngine to forecast next-day annual volatility and computes the edge vs market IV.

Usage:
  python main.py

Notes:
- Scanning the full S&P 500 can be slow and may hit rate limits; start with a small subset for testing
  by creating a tickers.txt file with one ticker per line.
- The script still relies on yfinance and arch; see requirements.txt for dependencies.
"""

from scanner_engine.data_fetcher import fetch_historical_prices, fetch_options_chain
from scanner_engine.math_engine import GARCHEngine

import datetime
import concurrent.futures
import time
import pandas as pd
import sys

# Configuration
EDGE_THRESHOLD = 0.04  # 4% in decimal
MAX_WORKERS = 6        # concurrency when scanning tickers (keep small to avoid rate limits)
SLEEP_BETWEEN_TICKERS = 0.2  # seconds
OPTION_PRICE_CAP = 1000.0  # only consider options priced below this
MIN_PRICE = 0.0001
MIN_HISTORY_DAYS = 100


def format_pct(x):
    try:
        return f"{float(x)*100:5.2f}%"
    except Exception:
        return "-"


def format_price(x):
    try:
        return f"${float(x):8.2f}"
    except Exception:
        return "-"


def print_report(recs):
    title = "🔥 OPTIONS RECOMMENDATIONS (S&P 500) — Contracts priced under $1000 🔥"
    print("\n" + title)
    print("=" * len(title))

    if not recs:
        print("No recommendations found that meet the edge threshold and price cap.")
        return

    cols = ["Ticker", "Type", "Strike", "Expiration", "Opt Price", "Market IV", "Predicted Vol", "Edge %"]
    widths = [8, 6, 10, 12, 12, 12, 14, 10]

    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))

    for r in recs:
        line = (
            f"{r['ticker']:<8} | {r['option_type']:<6} | {r['strike']:<10.2f} | {r['expiration']:<12} | "
            f"{format_price(r['opt_price']):<12} | {format_pct(r['market_iv']):<12} | {format_pct(r['pred_vol']):<14} | {r['edge']*100:6.2f}%"
        )
        print(line)


def load_watchlist(path="tickers.txt"):
    # If user provided tickers.txt, use it (one ticker per line).
    try:
        with open(path, "r") as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
            if tickers:
                print(f"Loaded {len(tickers)} tickers from {path}.")
                return tickers
    except FileNotFoundError:
        pass

    # Otherwise fetch S&P 500 symbols from Wikipedia
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        symbols = df['Symbol'].tolist()
        # Yahoo uses '-' instead of '.' for certain tickers (e.g. BRK.B -> BRK-B)
        symbols = [s.replace('.', '-').upper() for s in symbols]
        print(f"Loaded {len(symbols)} S&P 500 tickers.")
        return symbols
    except Exception as e:
        print(f"Failed to fetch S&P 500 list: {e}")
        return []


def option_mid_price(row):
    # Prefer mid=(bid+ask)/2 if both present and positive; fallback to lastPrice
    bid = row.get('bid') if 'bid' in row else None
    ask = row.get('ask') if 'ask' in row else None
    last = row.get('lastPrice') if 'lastPrice' in row else None

    try:
        bid = float(bid) if bid is not None and bid == bid else None
    except Exception:
        bid = None
    try:
        ask = float(ask) if ask is not None and ask == ask else None
    except Exception:
        ask = None
    try:
        last = float(last) if last is not None and last == last else None
    except Exception:
        last = None

    if bid and ask and ask > 0 and bid >= 0:
        mid = (bid + ask) / 2.0
        return mid
    if last and last > 0:
        return last
    if ask and ask > 0:
        return ask
    if bid and bid > 0:
        return bid
    return None


def scan_ticker(ticker):
    recs = []
    today = datetime.datetime.utcnow().date()

    try:
        prices = fetch_historical_prices(ticker)
        if prices is None or len(prices) < MIN_HISTORY_DAYS:
            # Not enough history
            return recs

        options_df = fetch_options_chain(ticker)
        if options_df is None or options_df.empty:
            return recs

        g = GARCHEngine(prices)
        pred_vol = g.forecast_next_day_annual_vol()
        if pred_vol is None:
            return recs

        # Iterate all option rows (all expirations & strikes)
        for _, row in options_df.iterrows():
            market_iv = None
            # normalize possible column names
            for k in ['impliedVolatility', 'impliedVol']:
                if k in row and row[k] == row[k]:
                    market_iv = row[k]
                    break
            if market_iv is None:
                continue

            opt_price = option_mid_price(row)
            if opt_price is None:
                continue
            if not (opt_price >= MIN_PRICE and opt_price < OPTION_PRICE_CAP):
                continue

            # compute edge (pred_vol and market_iv are decimals)
            try:
                edge = float(pred_vol) - float(market_iv)
            except Exception:
                continue

            if edge >= EDGE_THRESHOLD:
                exp = row.get('expiration')
                # normalize expiration string
                if isinstance(exp, str):
                    exp_str = exp
                else:
                    try:
                        exp_str = pd.to_datetime(exp).date().isoformat()
                    except Exception:
                        exp_str = str(exp)

                option_type = row.get('contractType') or row.get('type') or row.get('side') or row.get('option_type') or 'CALL'

                recs.append({
                    'ticker': ticker,
                    'option_type': 'Call' if str(option_type).lower().startswith('c') else 'Put',
                    'strike': float(row['strike']) if 'strike' in row and row['strike'] == row['strike'] else 0.0,
                    'expiration': exp_str,
                    'market_iv': float(market_iv),
                    'pred_vol': float(pred_vol),
                    'edge': float(edge),
                    'opt_price': float(opt_price),
                })

    except Exception as e:
        # swallow per-ticker errors to continue scanning others
        print(f"  Error scanning {ticker}: {e}")
    return recs


def main():
    watchlist = load_watchlist()
    if not watchlist:
        print("No tickers to scan. Exiting.")
        sys.exit(1)

    all_recs = []
    print(f"Starting scan of {len(watchlist)} tickers (concurrency={MAX_WORKERS})...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(scan_ticker, t): t for t in watchlist}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                recs = fut.result()
                if recs:
                    print(f"  Found {len(recs)} candidate(s) for {t}")
                all_recs.extend(recs)
            except Exception as e:
                print(f"  Error scanning {t}: {e}")
            # small sleep to be gentle on remote
            time.sleep(SLEEP_BETWEEN_TICKERS)

    # Sort by edge desc
    all_recs = sorted(all_recs, key=lambda x: x['edge'], reverse=True)
    print_report(all_recs)


if __name__ == '__main__':
    main()
