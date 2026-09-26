"""Short ATM straddle: how much risk does delta hedging remove, and what does it cost?

    uv run python scripts/run_hedging_experiment.py

All price paths are SIMULATED. The starting inputs come from the observed SPY
snapshot of 2026-09-25 (see reports/snapshot_summary.md) and are hard-coded
below, so the experiment reruns without the raw data.

Writes reports/hedging_results.csv, reports/hedging_results.md and
reports/figures/04_*.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_lab import hedging, plotting
from options_lab.simulate import diffusion_vol_for_total, price_paths

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"

# Observed on 2026-09-25 16:00 ET (reports/snapshot_summary.md)
SPOT = 771.30
STRIKE = 771.0
EXPIRY = 28 / 365
RATE = 0.04148          # ^IRX converted to continuous
IMPLIED_VOL = 0.1222    # Oct 23 771-strike implied vol, our calculation (call 12.20%, put 12.23%)

# Experiment design (assumptions)
CONTRACTS = 10                   # short 10 calls + 10 puts
N_STEPS = 260                    # 20 "days" x 13 half-hour decision points
STEPS_PER_DAY = 13
N_PATHS = 20_000
DEV_SEED, TEST_SEED = 20260926, 926  # dev paths pick the matched band; test paths give every reported number
COST_PER_SHARE = 0.01            # half the 1-cent SPY spread + ~0.5 cent fees
COST_SENSITIVITY = [0.0, 0.01, 0.05]
DRIFT = RATE                     # no view on direction

JUMPS = dict(jump_intensity=5.0, jump_mean=-0.015, jump_std=0.02)
SCENARIOS = {
    "base: realized = implied": dict(vol=IMPLIED_VOL),
    "realized 1.5x implied": dict(vol=1.5 * IMPLIED_VOL),
    "realized 0.75x implied": dict(vol=0.75 * IMPLIED_VOL),
    "jumps, same total variance": dict(vol=diffusion_vol_for_total(IMPLIED_VOL, **JUMPS), **JUMPS),
}
SCHEDULES = [1, 2, 4, 13, 26, 65]      # steps between rebalances (13 = daily, 65 = weekly)
BANDS = [10, 25, 50, 100, 200, 400]    # |net delta| trigger, in shares


def policies():
    return ([hedging.NoHedge()]
            + [hedging.EveryNSteps(n) for n in SCHEDULES]
            + [hedging.DeltaBand(w) for w in BANDS])


def describe(policy):
    if isinstance(policy, hedging.EveryNSteps):
        return "schedule", policy.n / STEPS_PER_DAY
    if isinstance(policy, hedging.DeltaBand):
        return "band", policy.width
    return "none", np.nan


def simulate(scenario, seed, n_paths=N_PATHS):
    rng = np.random.default_rng(seed)
    return price_paths(SPOT, DRIFT, t=EXPIRY, n_steps=N_STEPS, n_paths=n_paths, rng=rng, **scenario)


def evaluate(paths, times, straddle, policy, cost):
    result = hedging.run_straddle(paths, times, RATE, IMPLIED_VOL, straddle, policy, cost)
    family, setting = describe(policy)
    row = {"policy": policy.name, "family": family, "setting": setting, "cost_per_share": cost,
           **hedging.summarize(result.pnl),
           "mean_cost": result.cost.mean(), "mean_shares_traded": result.shares_traded.mean(),
           "mean_trades": result.n_trades.mean()}
    return row, result


def matched_band(times, straddle):
    """On DEV paths, the band whose average cost is closest to daily rebalancing."""
    paths = simulate(SCENARIOS["base: realized = implied"], DEV_SEED)
    daily_cost = evaluate(paths, times, straddle, hedging.EveryNSteps(13), COST_PER_SHARE)[0]["mean_cost"]
    costs = {w: evaluate(paths, times, straddle, hedging.DeltaBand(w), COST_PER_SHARE)[0]["mean_cost"] for w in BANDS}
    best = min(costs, key=lambda w: abs(costs[w] - daily_cost))
    return best, daily_cost, costs


def interval_label(days):
    hours = days * 6.5  # 13 half-hour steps per trading day
    return f"{days:g}d" if days >= 1 else (f"{hours:g}h" if hours >= 1 else f"{hours * 60:.0f}m")


def plot_risk_vs_cost(table):
    base = table[(table["scenario"] == "base: realized = implied") & (table["cost_per_share"] == COST_PER_SHARE)]
    fig, axes = plotting.figure(ncols=2, figsize=(11.5, 4.6))
    for ax, metric, label in [(axes[0], "std", "P&L standard deviation ($)"),
                              (axes[1], "expected_shortfall", "Average of worst 5% of outcomes ($)")]:
        for color, family, name in [(plotting.SERIES[0], "schedule", "Fixed schedule"),
                                    (plotting.SERIES[1], "band", "Delta band")]:
            rows = base[base["family"] == family].sort_values("mean_cost")
            ax.errorbar(rows["mean_cost"], rows[metric], yerr=2 * rows[f"{metric}_se"], color=color, lw=2,
                        marker="o", ms=7, mec=plotting.SURFACE, mew=1.5, capsize=0, label=name)
            last_x = 0
            for _, r in rows.iterrows():
                if r["mean_cost"] < 1.12 * last_x:
                    continue  # too close to the previous label
                last_x = r["mean_cost"]
                tag = interval_label(r["setting"]) if family == "schedule" else f"{r['setting']:g} sh"
                ax.annotate(tag, (r["mean_cost"], r[metric]), xytext=(0, 9 if family == "band" else -14),
                            textcoords="offset points", ha="center", fontsize=7, color=plotting.MUTED)
        ax.set_xscale("log")
        ax.set_xticks([20, 30, 50, 100])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Average transaction cost per straddle position ($, log scale)")
        ax.set_ylabel(label)
    none = base[base["family"] == "none"].iloc[0]
    axes[0].set_title(f"Risk left over (no hedge: {none['std']:,.0f})", loc="left", fontsize=10)
    axes[1].set_title(f"Tail loss (no hedge: {none['expected_shortfall']:,.0f})", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=9)
    plotting.title(fig, "Short 10-lot SPY straddle: risk remaining vs cost paid",
                   f"SIMULATED paths ({N_PATHS:,}, seed {TEST_SEED}); realized vol = implied {IMPLIED_VOL:.2%}; "
                   f"${COST_PER_SHARE}/share. Schedule labels = time between rebalances; band labels = shares of "
                   "net delta allowed. Error bars = ±2 Monte Carlo SE.")
    return fig


def plot_scenarios(pnl_by_scenario, band):
    fig, axes = plotting.figure(ncols=2, figsize=(11.5, 4.4))
    bins = np.linspace(-45_000, 25_000, 141)
    for color, (name, (unhedged, hedged)) in zip(plotting.SERIES, pnl_by_scenario.items()):
        axes[0].hist(unhedged, bins=bins, histtype="step", lw=2, color=color, density=True, label=name)
        axes[1].hist(hedged, bins=np.linspace(-20_000, 12_000, 161), histtype="step", lw=2,
                     color=color, density=True, label=name)
    axes[0].set_title("No hedge", loc="left", fontsize=10)
    axes[1].set_title(f"Delta band {band} shares (checked every half hour)", loc="left", fontsize=10)
    for ax in axes:
        ax.axvline(0, color=plotting.AXIS, lw=1)
        ax.set_xlabel("Terminal P&L ($)")
        ax.set_yticks([])
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    plotting.title(fig, "What delta hedging removes, and what it cannot",
                   f"SIMULATED. Same short 10-lot straddle, priced and hedged at {IMPLIED_VOL:.2%} vol in every "
                   f"scenario; only the simulated world changes. ${COST_PER_SHARE}/share costs. "
                   "x-axes are clipped; a few extreme losses fall outside.")
    return fig


def plot_convergence(times, straddle):
    rows = []
    for n in (1, 2, 4, 13, 26, 65, 130):
        paths = simulate(SCENARIOS["base: realized = implied"], TEST_SEED, n_paths=10_000)
        pnl = hedging.run_straddle(paths, times, RATE, IMPLIED_VOL, straddle, hedging.EveryNSteps(n)).pnl
        rows.append((n / STEPS_PER_DAY, pnl.std()))
    days, std = map(np.array, zip(*rows))
    fig, ax = plotting.figure(figsize=(7, 4.4))
    ax.plot(days, std, color=plotting.SERIES[0], lw=2, marker="o", ms=8, mec=plotting.SURFACE, mew=2,
            label="Simulated (no costs)")
    ax.plot(days, std[0] * np.sqrt(days / days[0]), color=plotting.MUTED, lw=1, ls="--", label="∝ √(interval)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Days between rebalances (log)")
    ax.set_ylabel("Std of hedged P&L ($, log)")
    ax.legend(frameon=False, fontsize=9)
    plotting.title(fig, "Check: with no costs and the right model, hedging error shrinks like √Δt",
                   "SIMULATED, 10,000 paths, realized = implied vol. A sanity check of the simulator, not a finding.")
    return fig, pd.DataFrame({"days_between": days, "pnl_std": std})


def plot_example_path(times, straddle, band):
    paths = simulate(SCENARIOS["base: realized = implied"], TEST_SEED, n_paths=200)
    k = int(np.argmax(np.abs(paths[:, -1] / SPOT - 1)))  # a path with a decent move
    path = paths[k:k + 1]
    days = times / EXPIRY * 20

    fig, axes = plotting.figure(nrows=3, figsize=(10, 8), sharex=True)
    axes[0].plot(days, path[0], color=plotting.INK, lw=1.5)
    axes[0].axhline(STRIKE, color=plotting.AXIS, lw=1, ls=":")
    axes[0].set_ylabel("SPY ($)")
    for color, policy in [(plotting.SERIES[0], hedging.EveryNSteps(13)),
                          (plotting.SERIES[1], hedging.DeltaBand(band)),
                          (plotting.SERIES[2], hedging.NoHedge())]:
        res = hedging.run_straddle(path, times, RATE, IMPLIED_VOL, straddle, policy, COST_PER_SHARE, record=True)
        marked = (res.cash[0, :-1] + res.holdings[0] * path[0, :-1]
                  + straddle.value(path[0, :-1], times[:-1], RATE, IMPLIED_VOL))
        marked = np.append(marked, res.pnl[0])
        label = {"every 13 steps": "Daily schedule"}.get(policy.name, policy.name.capitalize())
        axes[1].step(days[:-1], res.holdings[0], where="post", color=color, lw=2, label=label)
        axes[2].plot(days, marked, color=color, lw=2, label=f"{label}: final {res.pnl[0]:,.0f}")
    axes[1].set_ylabel("Shares of SPY held")
    axes[2].axhline(0, color=plotting.AXIS, lw=1)
    axes[2].set_ylabel("Marked P&L ($)")
    axes[2].set_xlabel("Trading days")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    plotting.title(fig, "One simulated path: how the hedge follows the market",
                   "SIMULATED path. Marked P&L = cash + stock + straddle valued with BSM at the implied vol.")
    return fig


def to_markdown(table, band, daily_dev_cost, dev_costs, convergence):
    show = ["policy", "mean", "mean_se", "std", "std_se", "tail_quantile", "expected_shortfall",
            "expected_shortfall_se", "mean_cost", "mean_shares_traded", "mean_trades"]
    lines = ["# Hedging experiment results (generated by scripts/run_hedging_experiment.py)", "",
             "All results are from SIMULATED price paths. Dollar amounts are per position "
             f"(short {CONTRACTS} calls + {CONTRACTS} puts, {CONTRACTS * 100} shares each leg).", "",
             f"- Start: SPY {SPOT}, strike {STRIKE}, {EXPIRY * 365:.0f} days, rate {RATE:.3%}, "
             f"priced and hedged at implied vol {IMPLIED_VOL:.2%} (observed 2026-09-25)",
             f"- Grid: {N_STEPS} steps ({STEPS_PER_DAY} per day), {N_PATHS:,} paths, test seed {TEST_SEED}",
             f"- Tail: `tail_quantile` = 5th percentile of P&L; `expected_shortfall` = mean of the worst 5%",
             f"- Matched band chosen on dev seed {DEV_SEED}: {band} shares (dev mean cost "
             f"{dev_costs[band]:,.0f} vs daily schedule {daily_dev_cost:,.0f})", ""]
    for (scenario, cost), rows in table.groupby(["scenario", "cost_per_share"], sort=False):
        lines += [f"## {scenario}, ${cost}/share", "", rows[show].round(1).to_markdown(index=False), ""]
    lines += ["## Convergence check (no costs, base scenario, 10,000 paths)", "",
              convergence.round(1).to_markdown(index=False), ""]
    return "\n".join(lines)


def main():
    times = np.linspace(0, EXPIRY, N_STEPS + 1)
    straddle = hedging.Straddle(STRIKE, EXPIRY, -CONTRACTS * 100)
    band, daily_dev_cost, dev_costs = matched_band(times, straddle)
    print(f"Matched band on dev paths: {band} shares")

    rows, pnl_by_scenario = [], {}
    for name, scenario in SCENARIOS.items():
        paths = simulate(scenario, TEST_SEED)
        costs = COST_SENSITIVITY if name.startswith("base") else [COST_PER_SHARE]
        for cost in costs:
            for policy in policies():
                row, result = evaluate(paths, times, straddle, policy, cost)
                rows.append({"scenario": name, **row})
                if cost == COST_PER_SHARE and policy.name in ("no hedge", f"band {band:g} sh"):
                    pnl_by_scenario.setdefault(name, []).append(result.pnl)
        print(f"done: {name}")
    table = pd.DataFrame(rows)

    FIGURES.mkdir(parents=True, exist_ok=True)
    convergence_fig, convergence = plot_convergence(times, straddle)
    for fname, fig in [("04_risk_vs_cost.png", plot_risk_vs_cost(table)),
                       ("04_scenarios.png", plot_scenarios(pnl_by_scenario, band)),
                       ("04_convergence.png", convergence_fig),
                       ("04_example_path.png", plot_example_path(times, straddle, band))]:
        fig.savefig(FIGURES / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    table.to_csv(ROOT / "reports" / "hedging_results.csv", index=False)
    report = to_markdown(table, band, daily_dev_cost, dev_costs, convergence)
    (ROOT / "reports" / "hedging_results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
