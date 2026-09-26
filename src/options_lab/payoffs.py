"""Option values at expiration, per share (one underlying unit).

All functions accept a single price or a NumPy array of prices, so one call
can evaluate a whole range of possible expiry prices at once (for plotting).
"""

import numpy as np


def call_payoff(spot_at_expiry, strike):
    """Value of a long call at expiration: the right to BUY at `strike`.

    Worth (spot - strike) if the stock finishes above the strike, otherwise 0.
    """
    return np.maximum(spot_at_expiry-strike, 0)


def put_payoff(spot_at_expiry, strike):
    """Value of a long put at expiration: the right to SELL at `strike`.

    Worth (strike - spot) if the stock finishes below the strike, otherwise 0.
    """
    return np.maximum(strike-spot_at_expiry,0)


def straddle_payoff(spot_at_expiry, strike):
    """Value of a long straddle at expiration: one call plus one put, same strike."""
    return(put_payoff(spot_at_expiry, strike) + call_payoff(spot_at_expiry, strike))


def position_profit(payoff, premium, quantity):
    """Profit per share at expiration for a position of `quantity` units.

    quantity = +1 for long (you paid the premium), -1 for short (you received it).
    Transaction costs are ignored here.
    """
    return quantity * (np.asarray(payoff) - premium)
