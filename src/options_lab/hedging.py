"""Delta-hedging simulator with a self-financing cash ledger.

Timeline on a grid t_0 < t_1 < ... < t_N (expiry):
    t_0      sell the options, receive the premium into cash
    t_i      observe S_i, decide whether to rebalance the stock hedge, trade at S_i,
             pay costs; the new holding is kept over (t_i, t_{i+1}]
    between  cash earns (or pays) the risk-free rate
    t_N      settle the options in cash at S_N and sell/buy back the hedge

The decision at t_i only uses S_i and earlier prices, so there is no lookahead.
Terminal P&L = final cash (we start with zero capital and borrow for the hedge).

Signs: `quantity` is in option units (shares), negative = short. A short
straddle has quantity -1000 for 10 contracts. Stock holding h is in shares.
"""

from dataclasses import dataclass

import numpy as np

from options_lab import bsm


@dataclass(frozen=True)
class Straddle:
    strike: float
    expiry: float          # years from t_0
    quantity: float        # option units per leg; negative = short; contracts x 100

    def value(self, spot, t_now, rate, vol):
        t_left = np.maximum(self.expiry - t_now, 0.0)
        per_unit = (bsm.call_price(spot, self.strike, t_left, rate, vol)
                    + bsm.put_price(spot, self.strike, t_left, rate, vol))
        return self.quantity * per_unit

    def delta(self, spot, t_now, rate, vol):
        """Position delta in shares."""
        t_left = np.maximum(self.expiry - t_now, 0.0)
        per_unit = (bsm.delta("call", spot, self.strike, t_left, rate, vol)
                    + bsm.delta("put", spot, self.strike, t_left, rate, vol))
        return self.quantity * per_unit

    def payoff(self, spot):
        return self.quantity * np.abs(spot - self.strike)


class NoHedge:
    name = "no hedge"

    def should_trade(self, step, net_delta):
        return np.zeros_like(net_delta, dtype=bool)


class EveryNSteps:
    """Rebalance to delta-neutral on a fixed schedule."""

    def __init__(self, n):
        self.n = n
        self.name = f"every {n} steps"

    def should_trade(self, step, net_delta):
        return np.full(net_delta.shape, step % self.n == 0)


class DeltaBand:
    """Rebalance to delta-neutral only when |net delta| exceeds `width` shares."""

    def __init__(self, width):
        self.width = width
        self.name = f"band {width:g} sh"

    def should_trade(self, step, net_delta):
        return np.abs(net_delta) > self.width


@dataclass
class HedgeResult:
    pnl: np.ndarray            # terminal P&L per path ($)
    cost: np.ndarray           # total transaction costs per path ($)
    shares_traded: np.ndarray  # total |shares| bought + sold, including liquidation
    n_trades: np.ndarray       # number of rebalances before expiry
    holdings: np.ndarray | None = None  # (n_paths, n_steps) shares held after each decision
    cash: np.ndarray | None = None      # (n_paths, n_steps + 1) cash after each step's trades


def run(paths, times, rate, premium, target_holding, payoff, policy, cost_per_share=0.0, record=False):
    """Simulate one hedging policy on the given paths.

    paths          (n_paths, N + 1) prices at `times`
    premium        cash received at t_0 (positive when selling options)
    target_holding function (spot_array, step) -> delta-neutral stock holding in shares
    payoff         function (spot_array) -> cash the option position receives at expiry
                   (negative for a short position)
    """
    n_paths, n_points = paths.shape
    n_steps = n_points - 1
    growth = np.exp(rate * np.diff(times))

    cash = np.full(n_paths, float(premium))
    held = np.zeros(n_paths)
    cost = np.zeros(n_paths)
    traded = np.zeros(n_paths)
    n_trades = np.zeros(n_paths, dtype=int)
    holdings = np.empty((n_paths, n_steps)) if record else None
    cash_path = np.empty((n_paths, n_points)) if record else None

    for i in range(n_steps):
        spot = paths[:, i]
        target = target_holding(spot, i)
        trade = policy.should_trade(i, held - target)  # held - target = net delta of book
        change = np.where(trade, target - held, 0.0)
        step_cost = cost_per_share * np.abs(change)

        cash -= change * spot + step_cost
        held += change
        cost += step_cost
        traded += np.abs(change)
        n_trades += trade & (change != 0)
        if record:
            holdings[:, i] = held
            cash_path[:, i] = cash
        cash *= growth[i]

    final_spot = paths[:, -1]
    close_cost = cost_per_share * np.abs(held)
    cash += held * final_spot - close_cost + payoff(final_spot)
    cost += close_cost
    traded += np.abs(held)
    if record:
        cash_path[:, -1] = cash
    return HedgeResult(cash, cost, traded, n_trades, holdings, cash_path)


def run_straddle(paths, times, rate, hedge_vol, straddle, policy, cost_per_share=0.0, record=False):
    """Sell `straddle` at its model price (vol = hedge_vol) and hedge it with BSM deltas."""
    premium = -straddle.value(paths[0, 0], 0.0, rate, hedge_vol)
    return run(
        paths, times, rate, premium,
        target_holding=lambda spot, i: -straddle.delta(spot, times[i], rate, hedge_vol),
        payoff=straddle.payoff,
        policy=policy, cost_per_share=cost_per_share, record=record,
    )


def summarize(pnl, tail=0.05, n_boot=300, rng=None):
    """Mean, std, lower-tail quantile and expected shortfall, with Monte Carlo standard errors.

    expected shortfall = average P&L of the worst `tail` fraction of paths (a loss is negative).
    Standard errors: analytic for the mean, bootstrap for the others.
    """
    rng = rng or np.random.default_rng(0)
    pnl = np.asarray(pnl)

    def stats(x):
        cutoff = np.quantile(x, tail)
        return np.array([x.std(ddof=1), cutoff, x[x <= cutoff].mean()])

    point = stats(pnl)
    boot = np.array([stats(pnl[rng.integers(0, len(pnl), len(pnl))]) for _ in range(n_boot)])
    se = boot.std(axis=0, ddof=1)
    return {
        "mean": pnl.mean(), "mean_se": pnl.std(ddof=1) / np.sqrt(len(pnl)),
        "std": point[0], "std_se": se[0],
        "tail_quantile": point[1], "tail_quantile_se": se[1],
        "expected_shortfall": point[2], "expected_shortfall_se": se[2],
    }
