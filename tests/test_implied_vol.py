import numpy as np
import pytest

from options_lab.bsm import price, vega
from options_lab.implied_vol import implied_vol

S, T, R, Q = 771.30, 28 / 365, 0.0415, 0.012


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [700.0, 771.0, 850.0])
@pytest.mark.parametrize("vol", [0.05, 0.12, 0.40, 1.5])
def test_round_trip(kind, strike, vol):
    # Numerical check only: recovering a vol we put in says nothing about markets.
    quote = price(kind, S, strike, T, R, vol, Q)
    recovered = implied_vol(quote, kind, S, strike, T, R, Q)
    assert price(kind, S, strike, T, R, recovered, Q) == pytest.approx(quote, abs=1e-8)
    if vega(S, strike, T, R, vol, Q) < 1e-3:
        pytest.skip("price barely depends on vol here, so the vol itself is not identifiable")
    assert recovered == pytest.approx(vol, abs=1e-6)


def test_below_intrinsic_has_no_iv():
    intrinsic_pv = S * np.exp(-Q * T) - 700 * np.exp(-R * T)
    assert np.isnan(implied_vol(intrinsic_pv - 0.50, "call", S, 700.0, T, R, Q))


def test_above_upper_bound_has_no_iv():
    assert np.isnan(implied_vol(S + 1, "call", S, 771.0, T, R, Q))


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.nan])
def test_nonsense_prices_have_no_iv(bad_price):
    assert np.isnan(implied_vol(bad_price, "put", S, 771.0, T, R, Q))


def test_expired_has_no_iv():
    assert np.isnan(implied_vol(5.0, "call", S, 771.0, 0.0, R, Q))
