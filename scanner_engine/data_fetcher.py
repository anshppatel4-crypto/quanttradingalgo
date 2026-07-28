"""scanner_engine/data_fetcher.py

Functions to fetch historical prices and options chain data using yfinance.
Cleans options by removing illiquid contracts (bid == 0 or bid-ask spread > 15%).
"""

import yfinance as yf
import pandas as pd
import numpy as np

def fetch_historical_prices(ticker: str, lookback_days: int = 500) -> pd.Series:
    """Fetch the last `lookback_days` trading daily close prices for `ticker`.

    Returns a pandas Series indexed by date with the Close prices.
    """
    tk = yf.Ticker(ticker)

    # Request a generous range and then take the last `lookback_days` trading days
    hist = tk.history(period="1500d", interval="1d", actions=False)
    if hist is None or hist.empty:
        return None

    closes = hist['Close'].dropna()
    if closes.empty:
        return None

    # Keep last `lookback_days` trading days
    closes = closes.tail(lookback_days)
    return closes

def _clean_options_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove illiquid contracts: bid == 0 or bid-ask spread wider than 15%.

    The spread percentage is computed as (ask - bid) / ask. Contracts with ask <= 0 are dropped.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # Ensure bid/ask columns exist
    for c in ['bid', 'ask']:
        if c not in df.columns:
            df[c] = np.nan

    # Drop rows where bid is zero or missing
    df = df[df['bid'].fillna(0) > 0]

    # Drop rows where ask is missing or ask <= 0
    df = df[df['ask'].fillna(0) > 0]

    # Compute spread percent
    df['spread_pct'] = (df['ask'] - df['bid']) / df['ask']

    df = df[df['spread_pct'] <= 0.15]

    # Cleanups
    df = df.drop(columns=['spread_pct'], errors='ignore')

    return df

def fetch_options_chain(ticker: str) -> pd.DataFrame:
    """Fetch the full options chain for `ticker` and return a single DataFrame with both calls and puts.

    The returned DataFrame includes columns: contractSymbol, strike, lastPrice, bid, ask, impliedVolatility, expiration, contractType
    """
    tk = yf.Ticker(ticker)

    try:
        expirations = tk.options
    except Exception:
        expirations = None

    if not expirations:
        return pd.DataFrame()

    rows = []
    for exp in expirations:
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue

        calls = oc.calls.copy()
        puts = oc.puts.copy()

        if not calls.empty:
            calls['expiration'] = exp
            calls['contractType'] = 'CALL'
            calls = _clean_options_df(calls)
            if not calls.empty:
                rows.append(calls)

        if not puts.empty:
            puts['expiration'] = exp
            puts['contractType'] = 'PUT'
            puts = _clean_options_df(puts)
            if not puts.empty:
                rows.append(puts)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True, sort=False)

    # Normalize implied volatility column name
    if 'impliedVol' in df.columns and 'impliedVolatility' not in df.columns:
        df['impliedVolatility'] = df['impliedVol']

    # Keep only relevant columns (if available)
    keep_cols = [c for c in ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility', 'expiration', 'contractType'] if c in df.columns]
    df = df[keep_cols]

    # Ensure strike numeric
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')

    # Drop rows with invalid strike
    df = df[df['strike'].notna()]

    # Sort
    df = df.sort_values(['expiration', 'strike']).reset_index(drop=True)

    return df
