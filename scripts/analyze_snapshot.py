"""Quote quality, implied-vol smile and term structure for one saved snapshot.

    uv run python scripts/analyze_snapshot.py [path/to/snapshot.csv]

Defaults to the most recent snapshot in data/raw/. Writes figures to
reports/figures/ and a summary table to reports/snapshot_summary.md.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_lab import chain, plotting

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"
SMILE_RANGE = 0.12  # plot log-moneyness within +/-12%


def plot_smiles(quotes, forwards, inputs):
    fig, ax = plotting.figure(figsize=(8.5, 5))
    for color, expiry in zip(plotting.SERIES, forwards["expiration"]):
        smile = quotes[(quotes["expiration"] == expiry) & quotes["otm"]].dropna(subset=["iv"])
        smile = smile[smile["log_moneyness"].abs() <= SMILE_RANGE].sort_values("log_moneyness")
        days = round(smile["t"].iloc[0] * 365)
        ax.plot(smile["log_moneyness"] * 100, smile["iv"] * 100, color=color, lw=2,
                label=f"{expiry} ({days}d)")
    ax.axvline(0, color=plotting.AXIS, lw=1)
    ax.set_xlabel("Strike vs forward, ln(K/F) (%)   ← OTM puts | OTM calls →")
    ax.set_ylabel("Implied volatility (%)")
    ax.legend(frameon=False, fontsize=9)
    plotting.title(fig, "SPY implied-volatility smile by expiry",
                   _source_note(inputs, "OTM options only; our IV from bid/ask mid with parity-implied forward"))
    return fig


def plot_call_vs_put(quotes, expiry):
    """Same-strike call and put IVs: ours (parity forward) vs Yahoo's reported number."""
    near = quotes[(quotes["expiration"] == expiry) & (quotes["log_moneyness"].abs() <= 0.03)]
    wide = near.pivot_table(index="strike", columns="option_type", values=["iv", "impliedVolatility"])

    fig, axes = plotting.figure(ncols=2, figsize=(11, 4.4), sharey=True)
    for ax, column, name in [(axes[0], "impliedVolatility", "Yahoo's reported IV"),
                             (axes[1], "iv", "Our IV (parity-implied forward)")]:
        ax.plot(wide.index, wide[(column, "call")] * 100, color=plotting.SERIES[0], lw=2, label="Call")
        ax.plot(wide.index, wide[(column, "put")] * 100, color=plotting.SERIES[1], lw=2, label="Put")
        gap = (wide[(column, "call")] - wide[(column, "put")]).abs().median() * 100
        ax.set_title(f"{name}\nmedian |call − put| gap: {gap:.2f} vol pts", loc="left", fontsize=10)
        ax.set_xlabel("Strike ($)")
    axes[0].set_ylabel("Implied volatility (%)")
    axes[0].legend(frameon=False, fontsize=9)
    plotting.title(fig, f"Same strike, same expiry ({expiry}): call and put should imply the same vol",
                   "Put–call parity links the two prices, so a European model gives them one vol. "
                   "Yahoo's gap disappears once the forward is taken from the quotes.")
    return fig


def plot_term_structure(forwards, inputs):
    fig, ax = plotting.figure(figsize=(7, 4.2))
    days = forwards["t"] * 365
    ax.plot(days, forwards["atm_vol"] * 100, color=plotting.SERIES[0], lw=2, marker="o", ms=8,
            mec=plotting.SURFACE, mew=2)
    for d, v, e in zip(days, forwards["atm_vol"] * 100, forwards["expiration"]):
        ax.annotate(f"{v:.1f}%\n{e}", (d, v), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=plotting.INK)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("ATM implied volatility (%)")
    ax.set_ylim(forwards["atm_vol"].min() * 100 - 1.5, forwards["atm_vol"].max() * 100 + 2)
    plotting.title(fig, "SPY at-the-money implied vol term structure", _source_note(inputs, "ATM = interpolated at the forward"))
    return fig


def _source_note(inputs, extra):
    return (f"Observed Yahoo Finance quotes, valued at {inputs['as_of']:%Y-%m-%d %H:%M} ET "
            f"(SPY {inputs['spot']:.2f}, r {inputs['rate']:.2%}). {extra}.")


def summary_table(quotes, forwards, inputs):
    flags = [c for c in quotes if c.startswith("flag_")]
    counts = quotes.groupby("expiration").agg(
        contracts=("strike", "size"),
        **{c.removeprefix("flag_"): (c, "sum") for c in flags},
        usable=("usable", "sum"),
        with_iv=("iv", "count"),
    ).reset_index()
    counts["usable_but_no_iv"] = counts["usable"] - counts["with_iv"]
    table = counts.merge(forwards[["expiration", "t", "pairs_used", "forward", "forward_iqr", "atm_vol"]])
    table["days"] = (table.pop("t") * 365).round(1)

    same_strike = quotes[quotes["log_moneyness"].abs() <= 0.03].pivot_table(
        index=["expiration", "strike"], columns="option_type", values=["iv", "impliedVolatility"]).dropna()
    gaps = pd.DataFrame({
        "ours": (same_strike[("iv", "call")] - same_strike[("iv", "put")]).abs(),
        "yahoo": (same_strike[("impliedVolatility", "call")] - same_strike[("impliedVolatility", "put")]).abs(),
    }).groupby("expiration").median() * 100

    lines = [
        "# Snapshot summary (generated by scripts/analyze_snapshot.py)",
        "",
        f"- Source: {inputs['meta']['source']}",
        f"- Retrieved: {inputs['meta']['retrieved_at_new_york']} (market open: {inputs['meta']['market_open_at_retrieval']})",
        f"- Valued as of: {inputs['as_of']:%Y-%m-%d %H:%M %Z}; SPY {inputs['spot']:.2f} "
        f"(1-min bar at {inputs['meta']['underlying_price_bar_time_utc']})",
        f"- Rate: ^IRX {inputs['meta']['short_rate_proxy']['last_close_percent']:.3f}% discount yield "
        f"→ {inputs['rate']:.4%} continuous",
        "",
        "## Quote quality and forwards by expiry",
        "",
        table.round(4).to_markdown(index=False),
        "",
        "## Median |call IV − put IV| at the same strike, within ±3% of the forward (vol points)",
        "",
        gaps.round(2).to_markdown(),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", nargs="?", type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot or sorted((ROOT / "data" / "raw").glob("*_chain_*.csv"))[-1]

    quotes, forwards, inputs = chain.analyze(snapshot)
    forwards["atm_vol"] = [chain.atm_vol(quotes, e) for e in forwards["expiration"]]
    focus = forwards.iloc[(forwards["t"] * 365 - 28).abs().argmin()]["expiration"]

    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, fig in [("03_iv_smile.png", plot_smiles(quotes, forwards, inputs)),
                      ("03_call_put_iv_gap.png", plot_call_vs_put(quotes, focus)),
                      ("03_term_structure.png", plot_term_structure(forwards, inputs))]:
        fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    summary = summary_table(quotes, forwards, inputs)
    (ROOT / "reports" / "snapshot_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
