import numpy as np

from options_lab.payoffs import call_payoff, position_profit, put_payoff, straddle_payoff


def test_call_payoff_above_at_and_below_strike():
    assert call_payoff(800, 767) == 33
    assert call_payoff(767, 767) == 0
    assert call_payoff(740, 767) == 0  # never negative: you just don't exercise


def test_put_payoff_above_at_and_below_strike():
    assert put_payoff(740, 767) == 27
    assert put_payoff(767, 767) == 0
    assert put_payoff(800, 767) == 0


def test_payoffs_work_on_arrays():
    spots = np.array([740.0, 767.0, 800.0])
    np.testing.assert_allclose(call_payoff(spots, 767), [0, 0, 33])
    np.testing.assert_allclose(put_payoff(spots, 767), [27, 0, 0])
    np.testing.assert_allclose(straddle_payoff(spots, 767), [27, 0, 33])


def test_call_minus_put_equals_spot_minus_strike():
    # At expiry, long call + short put (same strike) behaves exactly like owning
    # the stock and owing the strike, for every possible expiry price.
    spots = np.linspace(500, 1000, 101)
    np.testing.assert_allclose(call_payoff(spots, 767) - put_payoff(spots, 767), spots - 767)


def test_short_straddle_warmup_numbers():
    spots = np.array([767.0, 800.0, 740.0, 747.0, 787.0])
    profit = position_profit(straddle_payoff(spots, 767), premium=20, quantity=-1)
    np.testing.assert_allclose(profit, [20, -13, -7, 0, 0])
