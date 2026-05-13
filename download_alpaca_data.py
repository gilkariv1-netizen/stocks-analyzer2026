"""
One-time script to download historical 1-minute bars (including extended hours)
from Alpaca and save locally as Parquet.

Run once:
    python3 download_alpaca_data.py

After that, app.py will read from data/SOXL_1min.parquet automatically.
"""

import os
import sys
from datetime import datetime
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

TICKER     = "SOXL"
START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime.now()
OUT_DIR    = "data"
OUT_FILE   = f"{OUT_DIR}/{TICKER}_1min.parquet"

# Paste your keys here (or set as env vars ALPACA_KEY / ALPACA_SECRET)
API_KEY    = os.environ.get("ALPACA_KEY",    "YOUR_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET", "YOUR_SECRET_KEY")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "YOUR_API_KEY" in API_KEY:
        print("❌  Please set your Alpaca API key at the top of this file.")
        print("    Get one free at: https://alpaca.markets → Paper Trading → API Keys")
        sys.exit(1)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        print("❌  alpaca-py not installed. Run: python3 -m pip install alpaca-py")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

    print(f"Downloading {TICKER}  {START_DATE.date()} → {END_DATE.date()}")
    print("This may take a few minutes for multi-year 1-min data…\n")

    request = StockBarsRequest(
        symbol_or_symbols=TICKER,
        timeframe=TimeFrame.Minute,
        start=START_DATE,
        end=END_DATE,
        feed="iex",        # free tier — try this first
        adjustment="split",
    )

    try:
        bars = client.get_stock_bars(request)
        df = bars.df
    except Exception as e:
        print(f"❌  IEX feed failed: {e}")
        print("\n💡  SOXL trades on NYSE Arca, not IEX.")
        print("    Options:")
        print("    A) Upgrade to Alpaca 'Algo Trader Plus' ($9/mo) and change feed='sip'")
        print("    B) Use Polygon.io free tier instead (2 years of data)")
        print("       → run: python3 download_polygon_data.py  (coming next)")
        sys.exit(1)

    # Flatten MultiIndex (symbol, timestamp) → regular columns
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()

    # Normalize timestamp to Eastern Time
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").tz_convert("America/New_York")

    # Summary
    total   = len(df)
    regular = df["timestamp"].dt.time.between(
        pd.Timestamp("09:30").time(),
        pd.Timestamp("16:00").time()
    )
    ext_count = (~regular).sum()

    print(f"✅  Downloaded {total:,} bars")
    print(f"    Regular session : {regular.sum():,} bars")
    print(f"    Extended hours  : {ext_count:,} bars  ({ext_count/total*100:.1f}%)")
    print(f"    Date range      : {df['timestamp'].dt.date.min()} → {df['timestamp'].dt.date.max()}")

    df.to_parquet(OUT_FILE, index=False)
    print(f"\n💾  Saved to {OUT_FILE}")
    print("    You can now use extended hours mode in app.py")


if __name__ == "__main__":
    main()
