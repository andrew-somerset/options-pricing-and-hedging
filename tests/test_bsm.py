import numpy as np
import pytest

from options_lab import bsm
from options_lab.payoffs import call_payoff, put_payoff

S, K, T, R, VOL, Q = 771.30, 771.0, 28 / 365, 0.0415, 0.12, 0.012


def test_textbook_example():
    # Hull, Options Futures and Other Derivatives, Example 15.6: S=42, K=40, r=10%, vol=20%, T=0.5
    assert bsm.call_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=0.005)
    assert bsm.put_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("strike", [600.0, 750.0, 771.0, 800.0, 950.0])
@pytest.mark.parametrize("t", [1 / 365, 28 / 365, 1.0])
def test_put_call_parity(strike, t):
    lhs = bsm.call_price(S, strike, t, R, VOL, Q) - bsm.put_price(S, strike, t, R, VOL, Q)
    rhs = S * np.exp(-Q * t) - strike * np.exp(-R * t)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_prices_within_no_arbitrage_bounds():
    strikes = np.linspace(500, 1100, 61)
    call = bsm.call_price(S, strikes, T, R, VOL, Q)
    put = bsm.put_price(S, strikes, T, R, VOL, Q)
    spot_pv, strike_pv = S * np.exp(-Q * T), strikes * np.exp(-R * T)
    assert np.all(call >= np.maximum(spot_pv - strike_pv, 0) - 1e-12)
    assert np.all(call <= spot_pv)
    assert np.all(put >= np.maximum(strike_pv - spot_pv, 0) - 1e-12)
    assert np.all(put <= strike_pv)
    assert np.all(np.diff(call) <= 0)  # higher strike, cheaper call


def test_expiry_equals_payoff():
    spots = np.array([700.0, 771.0, 800.0])
    np.testing.assert_allclose(bsm.call_price(spots, K, 0.0, R, VOL), call_payoff(spots, K))
    np.testing.assert_allclose(bsm.put_price(spots, K, 0.0, R, VOL), put_payoff(spots, K))


def test_zero_vol_is_discounted_forward_payoff():
    expected = max(S * np.exp(-Q * T) - 700 * np.exp(-R * T), 0)
    assert bsm.call_price(S, 700.0, T, R, 0.0, Q) == pytest.approx(expected)
    assert bsm.put_price(S, 700.0, T, R, 0.0, Q) == pytest.approx(0.0)


def test_price_increases_with_vol():
    vols = np.linspace(0.05, 0.8, 16)
    assert np.all(np.diff(bsm.call_price(S, K, T, R, vols, Q)) > 0)


@pytest.mark.parametrize("bad", [dict(spot=-1.0), dict(strike=0.0), dict(t=-0.1), dict(vol=-0.2)])
def test_invalid_inputs_raise(bad):
    args = dict(spot=S, strike=K, t=T, rate=R, vol=VOL) | bad
    with pytest.raises(ValueError):
        bsm.call_price(**args)


# Analytic Greeks against central finite differences, away from expiry.
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [720.0, 771.0, 830.0])
def test_greeks_match_finite_differences(kind, strike):
    h_s, h_v, h_t = 0.01, 1e-5, 1e-6
    p = lambda s=S, v=VOL, t=T: bsm.price(kind, s, strike, t, R, v, Q)

    fd_delta = (p(s=S + h_s) - p(s=S - h_s)) / (2 * h_s)
    fd_gamma = (p(s=S + h_s) - 2 * p() + p(s=S - h_s)) / h_s**2
    fd_vega = (p(v=VOL + h_v) - p(v=VOL - h_v)) / (2 * h_v)
    fd_theta = -(p(t=T + h_t) - p(t=T - h_t)) / (2 * h_t)  # time passing = t shrinking

    assert bsm.delta(kind, S, strike, T, R, VOL, Q) == pytest.approx(fd_delta, rel=1e-5)
    assert bsm.gamma(S, strike, T, R, VOL, Q) == pytest.approx(fd_gamma, rel=1e-3)
    assert bsm.vega(S, strike, T, R, VOL, Q) == pytest.approx(fd_vega, rel=1e-5)
    assert bsm.theta(kind, S, strike, T, R, VOL, Q) == pytest.approx(fd_theta, rel=1e-4)


def test_atm_straddle_is_roughly_delta_neutral_and_long_gamma():
    straddle_delta = bsm.delta("call", S, K, T, R, VOL) + bsm.delta("put", S, K, T, R, VOL)
    assert abs(straddle_delta) < 0.1
    assert bsm.gamma(S, K, T, R, VOL) > 0
    assert bsm.theta("call", S, K, T, R, VOL) + bsm.theta("put", S, K, T, R, VOL) < 0
