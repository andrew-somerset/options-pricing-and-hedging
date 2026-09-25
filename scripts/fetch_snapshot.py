"""Save one timestamped option-chain snapshot from Yahoo Finance (via yfinance).

Run during US market hours (9:30-16:00 US/Eastern) so bid/ask quotes are live:

    uv run python scripts/fetch_snapshot.py

Writes two files to data/raw/ (never overwritten; data/ is not committed to git):
    <ticker>_chain_<UTC time>.csv   one row per contract (calls and puts)
    <ticker>_chain_<UTC time>.json  metadata: when, what, and caveats

yfinance is an unofficial library, not endorsed by Yahoo, for personal use.
Fetch once and reuse the saved files rather than re-downloading.
"""

import argparse
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

NEW_YORK = ZoneInfo("America/New_York")
# Aim for one expiration near each of these horizons (calendar days from today).
TARGET_DAYS = [7, 30, 60, 90]


def market_is_open(now_ny):
    """Rough regular-hours check. Ignores exchange holidays and early closes."""
    is_weekday = now_ny.weekday() < 5
    minutes = now_ny.hour * 60 + now_ny.minute
    return is_weekday and (9 * 60 + 30) <= minutes < 16 * 60


def pick_expirations(available, today, target_days):
    """For each target horizon, pick the listed expiration closest to it."""
    chosen = []
    for target in target_days:
        closest = min(available, key=lambda e: abs((date.fromisoformat(e) - today).days - target))
        if closest not in chosen:
            chosen.append(closest)
    return chosen


def fetch_with_retry(fn, what, attempts=3, wait_seconds=20):
    """Call fn(); if Yahoo rate-limits or fails, wait and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as err:
            if attempt == attempts:
                raise
            print(f"  {what} failed ({err!r}); retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--out", default=Path(__file__).resolve().parents[1] / "data" / "raw", type=Path)
    args = parser.parse_args()

    retrieved_utc = datetime.now(timezone.utc)
    retrieved_ny = retrieved_utc.astimezone(NEW_YORK)
    open_now = market_is_open(retrieved_ny)
    if not open_now:
        print("WARNING: outside regular US market hours. Quotes may be stale or zero;"
              " the metadata will record market_open_at_retrieval = false.")

    ticker = yf.Ticker(args.ticker)

    # Underlying price: the latest 1-minute bar gives a price WITH a timestamp.
    bars = fetch_with_retry(lambda: ticker.history(period="1d", interval="1m"), "underlying history")
    last_bar = bars.iloc[-1]
    underlying_price = float(last_bar["Close"])
    underlying_time = bars.index[-1].tz_convert("UTC").isoformat()

    available = fetch_with_retry(lambda: ticker.options, "expiration list")
    expirations = pick_expirations(available, retrieved_ny.date(), TARGET_DAYS)
    print(f"{args.ticker} last price {underlying_price:.2f} at {underlying_time}")
    print(f"Chosen expirations: {expirations}")

    frames = []
    for expiry in expirations:
        chain = fetch_with_retry(lambda: ticker.option_chain(expiry), f"chain {expiry}")
        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            table = table.copy()
            table.insert(0, "option_type", option_type)
            table.insert(1, "expiration", expiry)
            frames.append(table)
        time.sleep(1)  # be gentle with Yahoo
    quotes = pd.concat(frames, ignore_index=True)
    quotes["retrieved_at_utc"] = retrieved_utc.isoformat()

    # Inputs we will need for pricing later. Recorded raw, not yet interpreted.
    # ^IRX = 13-week US T-bill yield in percent (a short-rate proxy, discount basis).
    irx = fetch_with_retry(lambda: yf.Ticker("^IRX").history(period="5d"), "^IRX")
    dividends = fetch_with_retry(lambda: ticker.dividends, "dividends")
    last_year = dividends[dividends.index >= dividends.index.max() - pd.Timedelta(days=365)]

    stamp = retrieved_utc.strftime("%Y%m%dT%H%M%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{args.ticker.lower()}_chain_{stamp}.csv"
    meta_path = csv_path.with_suffix(".json")

    metadata = {
        "source": "Yahoo Finance via yfinance (unofficial; personal use)",
        "yfinance_version": yf.__version__,
        "ticker": args.ticker,
        "retrieved_at_utc": retrieved_utc.isoformat(),
        "retrieved_at_new_york": retrieved_ny.isoformat(),
        "market_open_at_retrieval": open_now,
        "underlying_price": underlying_price,
        "underlying_price_bar_time_utc": underlying_time,
        "expirations": expirations,
        "rows": {"total": len(quotes), "calls": int((quotes.option_type == "call").sum()),
                 "puts": int((quotes.option_type == "put").sum())},
        "short_rate_proxy": {"symbol": "^IRX", "last_close_percent": float(irx["Close"].iloc[-1]),
                             "as_of": irx.index[-1].isoformat()},
        "dividends_last_365d_per_share": float(last_year.sum()),
        "caveats": [
            "bid/ask have no quote timestamp; lastTradeDate is the last TRADE time, not the quote time",
            "impliedVolatility is Yahoo's own number, not computed by this project",
            "SPY options are American-style and physically settled; contract multiplier 100 shares",
            "midpoints are indicative marks, not guaranteed execution prices",
        ],
    }

    quotes.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(quotes)} contracts to {csv_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
