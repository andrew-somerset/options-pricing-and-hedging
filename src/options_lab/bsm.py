"""Black-Scholes-Merton prices and Greeks for European options.

Conventions (used everywhere in this project):
    spot, strike      price per share of the underlying
    t                 time to expiry in years, calendar days / 365
    rate              continuously compounded risk-free rate, annual decimal (0.04 = 4%)
    vol               annualized volatility, decimal (0.15 = 15%)
    div_yield         continuous dividend yield, annual decimal

Greeks are for ONE option on ONE share (multiply by quantity and the 100-share
contract multiplier yourself):
    delta  d(price)/d(spot)
    gamma  d(delta)/d(spot)
    vega   d(price)/d(vol) per 1.00 of vol; divide by 100 for "per vol point"
    theta  d(price)/d(calendar time) per year, so negative means the option
           loses value as time passes; divide by 365 for "per day"

Edge cases: at t = 0 prices equal the payoff; with vol = 0 the price is the
discounted payoff on the forward. Every function accepts NumPy arrays.
"""

import numpy as np
from scipy.stats import norm


def _check_inputs(spot, strike, t, vol):
    if np.any(np.asarray(spot) <= 0) or np.any(np.asarray(strike) <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(np.asarray(t) < 0):
        raise ValueError("time to expiry cannot be negative")
    if np.any(np.asarray(vol) < 0):
        raise ValueError("volatility cannot be negative")


def _d1_d2(spot, strike, t, rate, vol, div_yield):
    spot, strike, t, vol = np.broadcast_arrays(*map(np.asarray, (spot, strike, t, vol)))
    forward = spot * np.exp((rate - div_yield) * t)
    vol_sqrt_t = vol * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(forward / strike) / vol_sqrt_t + 0.5 * vol_sqrt_t
    # No uncertainty left: the option finishes in or out of the money for sure.
    d1 = np.where(vol_sqrt_t > 0, d1, np.where(forward > strike, np.inf, -np.inf))
    return d1, d1 - vol_sqrt_t, vol_sqrt_t


def _is_call(kind):
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    return kind == "call"


def price(kind, spot, strike, t, rate, vol, div_yield=0.0):
    _check_inputs(spot, strike, t, vol)
    d1, d2, _ = _d1_d2(spot, strike, t, rate, vol, div_yield)
    spot_pv = spot * np.exp(-div_yield * np.asarray(t))
    strike_pv = strike * np.exp(-rate * np.asarray(t))
    if _is_call(kind):
        return spot_pv * norm.cdf(d1) - strike_pv * norm.cdf(d2)
    return strike_pv * norm.cdf(-d2) - spot_pv * norm.cdf(-d1)


def call_price(spot, strike, t, rate, vol, div_yield=0.0):
    return price("call", spot, strike, t, rate, vol, div_yield)


def put_price(spot, strike, t, rate, vol, div_yield=0.0):
    return price("put", spot, strike, t, rate, vol, div_yield)


def delta(kind, spot, strike, t, rate, vol, div_yield=0.0):
    _check_inputs(spot, strike, t, vol)
    d1, _, _ = _d1_d2(spot, strike, t, rate, vol, div_yield)
    carry = np.exp(-div_yield * np.asarray(t))
    return carry * norm.cdf(d1) if _is_call(kind) else -carry * norm.cdf(-d1)


def gamma(spot, strike, t, rate, vol, div_yield=0.0):
    """Same for calls and puts."""
    _check_inputs(spot, strike, t, vol)
    d1, _, vol_sqrt_t = _d1_d2(spot, strike, t, rate, vol, div_yield)
    carry = np.exp(-div_yield * np.asarray(t))
    with np.errstate(divide="ignore", invalid="ignore"):
        g = carry * norm.pdf(d1) / (spot * vol_sqrt_t)
    return np.where(vol_sqrt_t > 0, g, 0.0)


def vega(spot, strike, t, rate, vol, div_yield=0.0):
    """Same for calls and puts. Per 1.00 of vol."""
    _check_inputs(spot, strike, t, vol)
    d1, _, _ = _d1_d2(spot, strike, t, rate, vol, div_yield)
    return spot * np.exp(-div_yield * np.asarray(t)) * norm.pdf(d1) * np.sqrt(t)


def theta(kind, spot, strike, t, rate, vol, div_yield=0.0):
    """Per year of calendar time. Undefined (nan) at t = 0."""
    _check_inputs(spot, strike, t, vol)
    d1, d2, vol_sqrt_t = _d1_d2(spot, strike, t, rate, vol, div_yield)
    t = np.asarray(t, dtype=float)
    spot_pv = spot * np.exp(-div_yield * t)
    strike_pv = strike * np.exp(-rate * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        decay = np.where(vol_sqrt_t > 0, -spot_pv * norm.pdf(d1) * vol / (2 * np.sqrt(t)), 0.0)
    if _is_call(kind):
        result = decay - rate * strike_pv * norm.cdf(d2) + div_yield * spot_pv * norm.cdf(d1)
    else:
        result = decay + rate * strike_pv * norm.cdf(-d2) - div_yield * spot_pv * norm.cdf(-d1)
    return np.where(t > 0, result, np.nan)
