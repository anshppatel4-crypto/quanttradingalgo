"""main.py
Updated entry point: scan tickers from tickers.txt and evaluate all option contracts priced under $10.

Key features:
- Uses ONLY tickers from tickers.txt (expects 104 tickers)
- For EACH ticker, exports top 5 options by estimated return %
- ONLY options with premium under $10 (opt_price < 10.0)
- Exports ALL tickers to results.csv
- Estimated return % = (pred_vol - market_iv) * sqrt(days_to_expiry / 365)
- Sorted by return % descending within each ticker
"""

from scanner_engine.data_fetcher import fetch_historical_prices, fetch_options_chain
from scanner_engine.math_engine import GARCHEngine

import datetime
import concurrent.futures
import time
import pandas as pd
import sys
import re

# Configuration
EDGE_THRESHOLD = 0.04  # 4% in decimal
MAX_WORKERS = 6        # concurrency when scanning tickers (keep small to avoid rate limits)
SLEEP_BETWEEN_TICKERS = 0.2  # seconds
OPTION_PRICE_CAP = 10.0  # only consider options priced below this ($10 premium)
MIN_PRICE = 0.0001
MIN_HISTORY_DAYS = 100
RESULTS_CSV = "results.csv"
TOP_N_PER_TICKER = 5  # Top 5 per ticker


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


def estimate_return_pct(pred_vol, market_iv, days_to_expiry):
    """Estimate percent return as volatility edge scaled by time.
    
    Return % = (pred_vol - market_iv) * sqrt(days_to_expiry / 365) * 100
    """
    try:
        pred_vol = float(pred_vol)
        market_iv = float(market_iv)
        days = max(1, int(days_to_expiry))
        
        # Volatility edge (in decimal, e.g. 0.15 for 15%)
        vol_edge = pred_vol - market_iv
        
        # Scale by time to expiration
        time_factor = (days / 365.0) ** 0.5
        
        # Estimated return as percentage
        est_ret = vol_edge * time_factor * 100
        return est_ret
    except Exception:
        return 0.0


def print_report(recs_by_ticker):
    """Print top 5 per ticker, grouped by ticker."""
    title = "🔥 TOP 5 OPTIONS PER TICKER (Premium < $10) — Estimated Return % 🔥"
    print("\n" + title)
    print("=" * len(title))

    if not recs_by_ticker:
        print("No recommendations found that meet the edge threshold and price cap.")
        return

    for ticker in sorted(recs_by_ticker.keys()):
        recs = recs_by_ticker[ticker]
        if not recs:
            continue

        print(f"\n--- {ticker} (Top {len(recs)}) ---")
        cols = ["Type", "Strike", "Expiration", "Premium", "Est Return %", "Market IV", "Predicted Vol", "Edge %"]
        widths = [6, 10, 12, 10, 14, 12, 14, 10]

        header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
        print(header)
        print("-" * len(header))

        for r in recs:
            line = (
                f"{r['option_type']:<6} | {r['strike']:<10.2f} | {r['expiration']:<12} | "
                f"{format_price(r['opt_price']):<10} | {r['est_return']:<14.2f} | {format_pct(r['market_iv']):<12} | "
                f"{format_pct(r['pred_vol']):<14} | {r['edge']*100:6.2f}%"
            )
            print(line)


def load_watchlist(path="tickers.txt"):
    """Load tickers from tickers.txt. File must exist."""
    try:
        raw = open(path, 'r', encoding='utf8').read()
        if raw and raw.strip():
            # Replace common escaped newline sequences with an actual newline
            cleaned = raw.replace('\\n', '\n').replace('\\N', '\n').replace('\\r\\n', '\n')
            # Replace commas/semicolons with newlines
            cleaned = re.sub(r'[;,]+', '\n', cleaned)
            # Extract tokens consisting of letters/digits/dot/dash
            tokens = re.findall(r'[A-Za-z0-9\.\-]+', cleaned)

            symbols = []
            for t in tokens:
                t = t.strip()
                if not t:
                    continue
                # If token looks artificially long (no separators), attempt to split
                if len(t) > 8:
                    # Find sequences of 1-5 uppercase letters/digits possibly with - (e.g. BRK-B)
                    parts = re.findall(r'[A-Z]{1,5}(?:-[A-Z0-9]{1,5})?', t.upper())
                    if parts:
                        symbols.extend(parts)
                        continue
                symbols.append(t.upper())

            # Normalize '.' -> '-' for Yahoo tickers and filter valid shapes
            normalized = []
            for s in symbols:
                s2 = s.replace('.', '-').upper()
                # Keep only tokens that look like tickers (letters/digits/dot/dash), length 1-8
                if re.fullmatch(r'[A-Z0-9\-\.]{1,8}', s2):
                    normalized.append(s2)

            if normalized:
                print(f"Loaded {len(normalized)} tickers from {path}.")
                return normalized
    except FileNotFoundError:
        print(f"ERROR: {path} not found. Please create it with your tickers.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: failed to parse {path}: {e}")
        sys.exit(1)

    print(f"ERROR: No tickers found in {path}")
    sys.exit(1)


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

    if bid is not None and ask is not None and ask > 0 and bid >= 0:
        mid = (bid + ask) / 2.0
        return mid
    if last is not None and last > 0:
        return last
    if ask is not None and ask > 0:
        return ask
    if bid is not None and bid > 0:
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
            # FILTER: Only options under $10 premium
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

                # Calculate days to expiry
                try:
                    exp_date = pd.to_datetime(exp).date()
                    days_to_exp = (exp_date - today).days
                    days_to_exp = max(1, days_to_exp)
                except Exception:
                    days_to_exp = 30

                # Calculate estimated return %
                est_ret = estimate_return_pct(pred_vol, market_iv, days_to_exp)

                recs.append({
                    'ticker': ticker,
                    'option_type': 'Call' if str(option_type).lower().startswith('c') else 'Put',
                    'strike': float(row['strike']) if 'strike' in row and row['strike'] == row['strike'] else 0.0,
                    'expiration': exp_str,
                    'market_iv': float(market_iv),
                    'pred_vol': float(pred_vol),
                    'edge': float(edge),
                    'opt_price': float(opt_price),
                    'est_return': est_ret,
                    'days_to_exp': days_to_exp,
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
    print(f"Filtering for options under ${OPTION_PRICE_CAP} premium...\n")

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

    # Group by ticker and get top 5 per ticker by return %
    recs_by_ticker = {}
    for rec in all_recs:
        ticker = rec['ticker']
        if ticker not in recs_by_ticker:
            recs_by_ticker[ticker] = []
        recs_by_ticker[ticker].append(rec)

    # Sort each ticker's options by est_return descending and keep top 5
    for ticker in recs_by_ticker:
        recs_by_ticker[ticker] = sorted(
            recs_by_ticker[ticker],
            key=lambda x: x['est_return'],
            reverse=True
        )[:TOP_N_PER_TICKER]

    print_report(recs_by_ticker)

    # Write ALL top 5 results (from all tickers) to CSV
    try:
        all_top_recs = []
        for ticker in sorted(recs_by_ticker.keys()):
            all_top_recs.extend(recs_by_ticker[ticker])

        if all_top_recs:
            df = pd.DataFrame(all_top_recs)
            # Order columns
            cols = ['ticker', 'option_type', 'strike', 'expiration', 'opt_price', 'market_iv', 'pred_vol', 'edge', 'est_return', 'days_to_exp']
            df = df[cols]
            # Sort by ticker, then by est_return desc
            df = df.sort_values(['ticker', 'est_return'], ascending=[True, False])
            df.to_csv(RESULTS_CSV, index=False)
            print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")
        else:
            print("\nNo candidates to write to CSV.")
    except Exception as e:
        print(f"Warning: failed to write {RESULTS_CSV}: {e}")


if __name__ == '__main__':
    main()
