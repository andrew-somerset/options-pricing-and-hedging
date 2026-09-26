import numpy as np
import pytest

from options_lab import hedging
from options_lab.bsm import call_price, put_price
from options_lab.simulate import diffusion_vol_for_total, price_paths


def scripted(targets):
    """Target holding that ignores the price and follows a fixed script, for hand checks."""
    return lambda spot, i: np.full(spot.shape, float(targets[i]))


def test_hand_calculated_path():
    # Sell for 10, buy 5 @100, buy 3 @110, sell 6 @108, close 2 @105, owe 5 at expiry.
    # Cost 0.10/share on every share traded, zero interest.
    #   cash: 10 - 500 - 0.5 = -490.5 -> -490.5 - 330 - 0.3 = -820.8
    #         -> -820.8 + 648 - 0.6 = -173.4 -> -173.4 + 210 - 0.2 - 5 = 31.4
    paths = np.array([[100.0, 110.0, 108.0, 105.0]])
    result = hedging.run(paths, np.array([0, 1, 2, 3.0]), rate=0.0, premium=10.0,
                         target_holding=scripted([5, 8, 2]), payoff=lambda s: np.full(s.shape, -5.0),
                         policy=hedging.EveryNSteps(1), cost_per_share=0.10)
    assert result.pnl[0] == pytest.approx(31.4)
    assert result.cost[0] == pytest.approx(1.6)
    assert result.shares_traded[0] == pytest.approx(16)
    assert result.n_trades[0] == 3


def test_cash_earns_interest():
    paths = np.full((1, 3), 100.0)
    result = hedging.run(paths, np.array([0, 0.5, 1.0]), rate=0.10, premium=100.0,
                         target_holding=scripted([0, 0]), payoff=lambda s: np.zeros(s.shape),
                         policy=hedging.NoHedge())
    assert result.pnl[0] == pytest.approx(100 * np.exp(0.10))


def test_rebalancing_at_unchanged_prices_creates_no_wealth():
    rng = np.random.default_rng(1)
    targets = rng.integers(-500, 500, size=50)
    paths = np.full((1, 51), 771.0)
    times = np.linspace(0, 1, 51)
    args = dict(rate=0.0, premium=0.0, target_holding=scripted(targets),
                payoff=lambda s: np.zeros(s.shape), policy=hedging.EveryNSteps(1))

    free = hedging.run(paths, times, **args)
    assert free.pnl[0] == pytest.approx(0.0, abs=1e-8)

    costly = hedging.run(paths, times, cost_per_share=0.01, **args)
    assert costly.pnl[0] == pytest.approx(-0.01 * costly.shares_traded[0])


def test_decisions_do_not_use_future_prices():
    rng = np.random.default_rng(2)
    times = np.linspace(0, 28 / 365, 21)
    paths = price_paths(771.3, 0.04, 0.12, times[-1], 20, 50, rng)
    shocked = paths.copy()
    shocked[:, -1] *= 1.3  # change only the final price
    straddle = hedging.Straddle(771.0, times[-1], -1000)
    for policy in (hedging.EveryNSteps(1), hedging.DeltaBand(50)):
        a = hedging.run_straddle(paths, times, 0.04, 0.12, straddle, policy, record=True)
        b = hedging.run_straddle(shocked, times, 0.04, 0.12, straddle, policy, record=True)
        np.testing.assert_array_equal(a.holdings, b.holdings)


def test_policies():
    net = np.array([-60.0, -10.0, 0.0, 30.0, 80.0])
    assert not hedging.NoHedge().should_trade(0, net).any()
    assert hedging.EveryNSteps(5).should_trade(10, net).all()
    assert not hedging.EveryNSteps(5).should_trade(11, net).any()
    assert hedging.DeltaBand(50).should_trade(3, net).tolist() == [True, False, False, False, True]


def test_straddle_value_delta_payoff_signs():
    short = hedging.Straddle(771.0, 28 / 365, -1000)
    model = call_price(771.3, 771.0, 28 / 365, 0.04, 0.12) + put_price(771.3, 771.0, 28 / 365, 0.04, 0.12)
    assert short.value(771.3, 0.0, 0.04, 0.12) == pytest.approx(-1000 * model)
    assert short.payoff(np.array([760.0, 771.0, 790.0])).tolist() == [-11000.0, 0.0, -19000.0]
    # short straddle: delta becomes negative when spot rises (you lose as it moves away)
    assert short.delta(800.0, 0.0, 0.04, 0.12) < 0 < short.delta(740.0, 0.0, 0.04, 0.12)


def test_zero_cost_hedging_error_shrinks_on_finer_grid():
    """Correctly specified model, no costs: mean P&L ~ 0 and dispersion falls ~ sqrt(dt)."""
    t, vol, rate = 28 / 365, 0.12, 0.04
    straddle = hedging.Straddle(771.0, t, -1000)
    stds = []
    for n_steps in (20, 320):
        rng = np.random.default_rng(3)
        times = np.linspace(0, t, n_steps + 1)
        paths = price_paths(771.3, rate, vol, t, n_steps, 4000, rng)
        pnl = hedging.run_straddle(paths, times, rate, vol, straddle, hedging.EveryNSteps(1)).pnl
        assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(len(pnl))
        stds.append(pnl.std())
    # 16x more steps -> about 4x smaller error in theory; allow slack
    assert stds[1] < 0.35 * stds[0]


def test_price_paths_mean_and_variance():
    rng = np.random.default_rng(4)
    t, drift = 0.5, 0.05
    diffusion = diffusion_vol_for_total(0.2, jump_intensity=5, jump_mean=-0.02, jump_std=0.03)
    paths = price_paths(100.0, drift, diffusion, t, 50, 100_000, rng,
                        jump_intensity=5, jump_mean=-0.02, jump_std=0.03)
    assert paths[:, 0].tolist() == [100.0] * 100_000
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(drift * t), rel=2e-3)
    assert np.log(paths[:, -1] / 100).var() == pytest.approx(0.2**2 * t, rel=0.03)
