"""main.py
Updated entry point: scan the S&P 500 (or tickers.txt) and evaluate all option contracts priced under $1000.

Improvements in this update:
- Robust tickers.txt parsing & normalization to handle literal "\\n" / "\\N" escapes, commas, semicolons,
  and some concatenated uppercase sequences (e.g. AAPLMSFTAMZN will be split into AAPL, MSFT, AMZN when possible).
- Validation of ticker tokens (keeps A-Z, 0-9, dot, dash) and normalizes '.' -> '-' for Yahoo format.
- Writes results.csv with candidate rows (columns: ticker, option_type, strike, expiration, opt_price, market_iv, pred_vol, edge, est_return).
- Keeps previous behavior (S&P fetch fallback, concurrency, option price cap, GARCH engine).
- UPDATED: Top 5 per option type, added QQQ/SPY/ODT, added estimated percent return.
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
OPTION_PRICE_CAP = 1000.0  # only consider options priced below this
MIN_PRICE = 0.0001
MIN_HISTORY_DAYS = 100
RESULTS_CSV = "results.csv"
TOP_N_PER_TYPE = 5  # Top 5 per option type


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


def estimate_return(opt_price, pred_vol, days_to_expiry):
    """Estimate percent return assuming volatility mean-reversion.
    
    Simple model: if predicted vol > market IV, option will gain value.
    Rough estimate: return ≈ (pred_vol - market_iv) * sqrt(days_to_expiry / 365)
    """
    try:
        opt_price = float(opt_price)
        pred_vol = float(pred_vol)
        if opt_price <= 0:
            return 0.0
        # Days to expiry as fraction of year
        t = max(1, days_to_expiry) / 365.0
        # Volatility edge annualized and discounted to time frame
        vol_edge = pred_vol * (t ** 0.5)
        # Rough option value change per 1% vol change ≈ vega (simplified)
        # Assume option delta ≈ 0.5 for ATM, use rough vega
        est_return = vol_edge * opt_price * 0.5  # very rough estimate
        return est_return / opt_price if opt_price > 0 else 0.0
    except Exception:
        return 0.0


def print_report(recs_by_type):
    """Print top 5 calls and top 5 puts separately."""
    title = "🔥 TOP OPTIONS RECOMMENDATIONS (QQQ, SPY, ODT + S&P 500) — Under $1000 🔥"
    print("\n" + title)
    print("=" * len(title))

    if not recs_by_type or all(not v for v in recs_by_type.values()):
        print("No recommendations found that meet the edge threshold and price cap.")
        return

    for option_type in ['Call', 'Put']:
        recs = recs_by_type.get(option_type, [])
        if not recs:
            continue

        print(f"\n--- TOP {len(recs)} {option_type.upper()}S ---")
        cols = ["Ticker", "Type", "Strike", "Expiration", "Opt Price", "Market IV", "Predicted Vol", "Edge %", "Est Return %"]
        widths = [8, 6, 10, 12, 12, 12, 14, 10, 12]

        header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
        print(header)
        print("-" * len(header))

        for r in recs:
            line = (
                f"{r['ticker']:<8} | {r['option_type']:<6} | {r['strike']:<10.2f} | {r['expiration']:<12} | "
                f"{format_price(r['opt_price']):<12} | {format_pct(r['market_iv']):<12} | {format_pct(r['pred_vol']):<14} | {r['edge']*100:6.2f}% | {r['est_return']*100:8.2f}%"
            )
            print(line)


def load_watchlist(path="tickers.txt"):
    """Load tickers from file if present, otherwise use default watchlist + S&P 500.
    
    Always includes: QQQ, SPY, ODT
    """
    default_tickers = ['QQQ', 'SPY', 'ODT']
    
    # If user provided tickers.txt, try to parse & normalize it
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
                # Add defaults if not already present
                for t in default_tickers:
                    if t not in normalized:
                        normalized.insert(0, t)
                print(f"Loaded {len(normalized)} tickers from {path} (normalized), including defaults.")
                return normalized
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Warning: failed to parse {path}: {e}")

    # Fallback: fetch S&P 500 symbols from Wikipedia + defaults
    print("Fetching S&P 500 tickers from Wikipedia...")
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-').upper() for s in symbols]
        # Add defaults at beginning
        for t in reversed(default_tickers):
            if t not in symbols:
                symbols.insert(0, t)
        print(f"Loaded {len(symbols)} tickers ({len(default_tickers)} defaults + S&P 500).")
        return symbols
    except Exception as e:
        print(f"Failed to fetch S&P 500 list: {e}")
        print(f"Using defaults only: {default_tickers}")
        return default_tickers


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

                # Calculate estimated return
                est_ret = estimate_return(opt_price, pred_vol, days_to_exp)

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

    # Split into calls and puts, keep top 5 each
    calls = [r for r in all_recs if r['option_type'] == 'Call'][:TOP_N_PER_TYPE]
    puts = [r for r in all_recs if r['option_type'] == 'Put'][:TOP_N_PER_TYPE]

    recs_by_type = {'Call': calls, 'Put': puts}
    print_report(recs_by_type)

    # Also write results.csv for later analysis
    try:
        top_recs = calls + puts
        if top_recs:
            df = pd.DataFrame(top_recs)
            # Order columns
            cols = ['ticker', 'option_type', 'strike', 'expiration', 'opt_price', 'market_iv', 'pred_vol', 'edge', 'est_return', 'days_to_exp']
            df = df[cols]
            df.to_csv(RESULTS_CSV, index=False)
            print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")
        else:
            print("\nNo candidates to write to CSV.")
    except Exception as e:
        print(f"Warning: failed to write {RESULTS_CSV}: {e}")


if __name__ == '__main__':
    main()
