"""Implied volatility: the vol that makes the Black-Scholes price match a quote.

A price only has an implied vol if it sits strictly between the model's
no-arbitrage bounds (vol -> 0 and vol -> infinity). Quotes outside those bounds
return nan instead of a forced, plausible-looking number.
"""

import numpy as np
from scipy.optimize import brentq

from options_lab.bsm import price

VOL_LOW, VOL_HIGH = 1e-4, 5.0


def price_bounds(kind, spot, strike, t, rate, div_yield=0.0):
    """(lower, upper) European price bounds under the model."""
    spot_pv = spot * np.exp(-div_yield * t)
    strike_pv = strike * np.exp(-rate * t)
    if kind == "call":
        return max(spot_pv - strike_pv, 0.0), spot_pv
    return max(strike_pv - spot_pv, 0.0), strike_pv


def implied_vol(option_price, kind, spot, strike, t, rate, div_yield=0.0, tol=1e-10):
    """Implied vol of one quote, or nan if none exists in [VOL_LOW, VOL_HIGH]."""
    if not np.isfinite(option_price) or t <= 0:
        return np.nan
    lower, upper = price_bounds(kind, spot, strike, t, rate, div_yield)
    if not lower < option_price < upper:
        return np.nan

    def gap(vol):
        return price(kind, spot, strike, t, rate, vol, div_yield) - option_price

    if gap(VOL_LOW) > 0 or gap(VOL_HIGH) < 0:
        return np.nan  # inside the bounds but outside the vol range we search
    return brentq(gap, VOL_LOW, VOL_HIGH, xtol=tol)
