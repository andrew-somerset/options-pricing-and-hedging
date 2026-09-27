# Options Pricing and Hedging

Selling an at-the-money SPY straddle: pricing it from real market quotes, then measuring how much of its risk delta hedging removes, what that costs, and what hedging cannot remove.

## Results

| | Finding |
|---|---|
| **Real SPY option chain** (2026-09-25 close) | Backing the forward price out of put–call parity makes same-strike call and put implied vols agree to within 0.2–0.4 vol points. Yahoo's reported IVs for the same contracts differ by 2–4 points. |
| **Delta hedging** (simulated) | Daily hedging of a short 10-lot straddle cut P&L standard deviation by 75% ($15.9k → $4.0k) for about $37 in trading costs. |
| **Band vs schedule** (simulated) | At matched cost, rebalancing only when net delta exceeds 200 shares gave 32% lower P&L std and a 38% smaller worst-5% loss than rebalancing daily. |
| **What hedging can't fix** (simulated) | If realized vol is 1.5× implied, the hedged position loses about $10.5k under *every* rebalancing rule. With jumps, the worst-5% loss stays around $21k even when hedging every 30 minutes. |

![Risk remaining vs cost paid](reports/figures/04_risk_vs_cost.png)

Full write-up with tables, charts and Monte Carlo standard errors: [reports/REPORT.md](reports/REPORT.md).

## How it works

1. **Pricing and Greeks.** European Black–Scholes–Merton prices, delta, gamma, vega and theta, and a bracketed implied-vol solver. Checked against a textbook example, put–call parity, no-arbitrage bounds, finite-difference Greeks and price → IV → price round trips (75 tests).
2. **Real snapshot.** One SPY option chain (1,124 contracts, 4 expiries). Bad quotes are flagged rather than silently dropped, forwards are backed out from put–call parity, and our own implied vols give the volatility smile and term structure.
3. **Hedging experiment.** A short straddle sold at the observed implied vol, then hedged on 20,000 simulated price paths with a self-financing cash ledger (premium, stock trades, interest, costs, settlement). Three policy families run on identical paths: no hedge, a fixed schedule, and a delta band. Stress tests use realized vol ≠ implied vol and jumps with the same total variance. The band setting compared against daily hedging was chosen on separate development paths.

## Limitations

- The hedging results come from simulated paths (constant-vol GBM and Merton jumps). They are not a historical backtest.
- SPY options are American-style and physically settled; the European model is an approximation.
- Flat-vol deltas, a uniform time grid, linear transaction costs, and the option trade at mid.
- An educational pricing and hedging study, not a market-making simulator: no queue position, adverse selection, customer flow or spread capture.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                          # pinned environment
uv run pytest                                    # 75 checks
uv run python scripts/run_hedging_experiment.py  # simulated hedging comparison, ~30 s, no data needed
uv run python scripts/fetch_snapshot.py          # save your own option-chain snapshot (US market hours)
uv run python scripts/analyze_snapshot.py        # smile, term structure, quote-quality table
```

## Layout

```
src/options_lab/   payoffs, bsm (prices + Greeks), implied_vol, chain (snapshot cleaning + IV),
                   simulate (price paths), hedging (cash ledger + policies), plotting
scripts/           fetch_snapshot, analyze_snapshot, run_hedging_experiment
tests/             pricing, IV, snapshot pipeline and hedging-ledger checks
notebooks/         01_straddle_payoff (payoff vs profit with real prices)
reports/           REPORT.md, generated tables and figures
```

## Data

Option quotes come from Yahoo Finance through the unofficial `yfinance` library, for personal research use. Raw quotes aren't committed; the repository contains only derived summaries and charts. Simulated results are labeled as simulated throughout.
