"""Clean one saved option-chain snapshot and compute our own implied vols.

Steps:
    1. load quotes + metadata, attach time to expiry
    2. flag bad quotes (flags are kept, rows are never silently dropped)
    3. back out a forward price per expiry from put-call parity
    4. implied vol for each usable quote, using that forward

Using the parity-implied forward avoids having to guess SPY's discrete dividends.
European formulas are an approximation here: SPY options are American-style.
"""

import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from options_lab.implied_vol import implied_vol

NEW_YORK = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)

MIN_MID = 0.05             # below this, the 1-cent tick is a large share of the price
MAX_RELATIVE_SPREAD = 0.5  # (ask - bid) / mid
PARITY_BAND = 0.05         # strikes within +/-5% of spot are used to estimate the forward


def tbill_discount_to_continuous(discount_yield_pct, days=91):
    """Convert a T-bill discount yield (e.g. ^IRX, in percent) to a continuous rate."""
    bill_price = 1 - discount_yield_pct / 100 * days / 360
    return -np.log(bill_price) / (days / 365)


def load_snapshot(csv_path):
    csv_path = Path(csv_path)
    quotes = pd.read_csv(csv_path)
    meta = json.loads(csv_path.with_suffix(".json").read_text())
    return quotes, meta


def valuation_time(meta):
    """Retrieval time, capped at that day's 16:00 ET close (after-hours quotes are closing quotes)."""
    retrieved = datetime.fromisoformat(meta["retrieved_at_utc"]).astimezone(NEW_YORK)
    close = datetime.combine(retrieved.date(), MARKET_CLOSE, tzinfo=NEW_YORK)
    return min(retrieved, close)


def years_to_expiry(expirations, as_of):
    """Calendar years from `as_of` to 16:00 ET on each expiration date."""
    expiry = pd.to_datetime(expirations).map(
        lambda d: datetime.combine(d.date(), MARKET_CLOSE, tzinfo=NEW_YORK))
    return np.array([(e - as_of).total_seconds() for e in expiry]) / (365 * 24 * 3600)


def flag_quotes(quotes):
    q = quotes.copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2
    q["spread"] = q["ask"] - q["bid"]
    q["flag_no_bid"] = ~(q["bid"] > 0)  # zero or missing
    q["flag_no_ask"] = ~(q["ask"] > 0)
    q["flag_crossed"] = q["ask"] < q["bid"]
    q["flag_tiny_price"] = q["mid"] < MIN_MID
    with np.errstate(divide="ignore", invalid="ignore"):
        q["flag_wide_spread"] = (q["spread"] / q["mid"]) > MAX_RELATIVE_SPREAD
    flags = ["flag_no_bid", "flag_no_ask", "flag_crossed", "flag_tiny_price", "flag_wide_spread"]
    q["usable"] = ~q[flags].any(axis=1)
    return q


def implied_forwards(quotes, spot, rate):
    """Forward price per expiry from put-call parity: C - P = D * (F - K).

    Uses usable call/put pairs at the same strike near the money and the T-bill
    rate for D. Returns one row per expiry with the median forward and its spread
    across strikes. `implied_div_yield` is the q that reproduces that forward.
    """
    rows = []
    for expiry, group in quotes[quotes["usable"]].groupby("expiration"):
        pairs = group.pivot_table(index="strike", columns="option_type", values="mid").dropna()
        pairs = pairs[np.abs(pairs.index / spot - 1) <= PARITY_BAND]
        t = group["t"].iloc[0]
        discount = np.exp(-rate * t)
        forwards = pairs.index + (pairs["call"] - pairs["put"]) / discount
        rows.append({
            "expiration": expiry,
            "t": t,
            "pairs_used": len(pairs),
            "forward": forwards.median(),
            "forward_iqr": forwards.quantile(0.75) - forwards.quantile(0.25),
        })
    fwd = pd.DataFrame(rows)
    fwd["implied_carry"] = np.log(fwd["forward"] / spot) / fwd["t"]  # r - q
    fwd["implied_div_yield"] = rate - fwd["implied_carry"]
    return fwd


def add_implied_vols(quotes, forwards, spot, rate):
    q = quotes.merge(forwards[["expiration", "forward", "implied_div_yield"]], on="expiration")
    q["log_moneyness"] = np.log(q["strike"] / q["forward"])
    q["otm"] = np.where(q["option_type"] == "call", q["strike"] >= q["forward"], q["strike"] < q["forward"])
    q["iv"] = [
        implied_vol(row.mid, row.option_type, spot, row.strike, row.t, rate, row.implied_div_yield)
        if row.usable else np.nan
        for row in q.itertuples()
    ]
    return q


def analyze(csv_path):
    """Run the full pipeline. Returns (quotes with IVs, forwards per expiry, inputs used)."""
    quotes, meta = load_snapshot(csv_path)
    as_of = valuation_time(meta)
    spot = meta["underlying_price"]
    rate = tbill_discount_to_continuous(meta["short_rate_proxy"]["last_close_percent"])

    quotes["t"] = years_to_expiry(quotes["expiration"], as_of)
    quotes = flag_quotes(quotes)
    forwards = implied_forwards(quotes, spot, rate)
    quotes = add_implied_vols(quotes, forwards, spot, rate)
    inputs = {"as_of": as_of, "spot": spot, "rate": rate, "meta": meta}
    return quotes, forwards, inputs


def atm_vol(quotes, expiry):
    """Implied vol at the forward, by linear interpolation of the OTM smile in log-moneyness."""
    smile = quotes[(quotes["expiration"] == expiry) & quotes["otm"]].dropna(subset=["iv"])
    smile = smile.sort_values("log_moneyness")
    return float(np.interp(0.0, smile["log_moneyness"], smile["iv"]))
