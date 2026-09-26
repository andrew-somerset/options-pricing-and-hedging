from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from options_lab import chain
from options_lab.bsm import price

NY = chain.NEW_YORK


def test_tbill_conversion():
    # 4.07% discount yield on a 91-day bill -> price 0.98971 -> about 4.15% continuous
    assert chain.tbill_discount_to_continuous(4.07) == pytest.approx(0.04148, abs=1e-5)


def test_after_hours_snapshot_is_valued_at_the_close():
    meta = {"retrieved_at_utc": "2026-09-25T20:14:52+00:00"}
    assert chain.valuation_time(meta) == datetime(2026, 9, 25, 16, 0, tzinfo=NY)


def test_years_to_expiry_close_to_close():
    as_of = datetime(2026, 9, 25, 16, 0, tzinfo=NY)
    assert chain.years_to_expiry(["2026-10-23"], as_of)[0] == pytest.approx(28 / 365)


def test_flags():
    quotes = pd.DataFrame({
        "bid": [1.00, 0.00, np.nan, 2.10, 0.01, 1.00],
        "ask": [1.10, 0.05, 0.10, 2.00, 0.02, 3.00],
    })
    q = chain.flag_quotes(quotes)
    assert q["usable"].tolist() == [True, False, False, False, False, False]
    assert q["flag_no_bid"].tolist() == [False, True, True, False, False, False]
    assert q["flag_crossed"].tolist() == [False, False, False, True, False, False]
    assert q["flag_tiny_price"].tolist() == [False, True, False, False, True, False]  # missing mid is caught by flag_no_bid
    assert q["flag_wide_spread"].tolist() == [False, True, False, False, True, True]


def synthetic_chain(spot, rate, div_yield, t, vol, strikes):
    """Model-generated quotes (SYNTHETIC, for testing only) with a 2-cent spread."""
    rows = []
    for kind in ("call", "put"):
        mids = price(kind, spot, strikes, t, rate, vol, div_yield)
        for k, m in zip(strikes, mids):
            rows.append({"option_type": kind, "expiration": "2026-10-23", "strike": k,
                         "bid": m - 0.01, "ask": m + 0.01, "t": t})
    return chain.flag_quotes(pd.DataFrame(rows))


def test_forward_and_iv_recovered_from_synthetic_quotes():
    spot, rate, q_true, t, vol = 771.3, 0.0415, 0.013, 28 / 365, 0.14
    strikes = np.arange(740.0, 805.0, 1.0)
    quotes = synthetic_chain(spot, rate, q_true, t, vol, strikes)

    fwd = chain.implied_forwards(quotes, spot, rate)
    assert fwd["forward"].iloc[0] == pytest.approx(spot * np.exp((rate - q_true) * t), abs=1e-6)
    assert fwd["implied_div_yield"].iloc[0] == pytest.approx(q_true, abs=1e-6)

    with_iv = chain.add_implied_vols(quotes, fwd, spot, rate)
    np.testing.assert_allclose(with_iv["iv"], vol, atol=1e-6)
    assert chain.atm_vol(with_iv, "2026-10-23") == pytest.approx(vol, abs=1e-6)
