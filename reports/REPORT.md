# Selling a SPY straddle: pricing, the real smile, and what hedging costs

**Question.** If I sell an at-the-money option position, what risks do I take on, how can I measure them, and how much does it cost to reduce the directional part?

**Setup.** A short 10-lot SPY straddle (10 calls + 10 puts, strike 771, expiring 2026-10-23), priced from a real option-chain snapshot taken after the close on 2026-09-25. The hedging results use **simulated** price paths that start from those observed inputs. Nothing here is a historical backtest.

| | Source |
|---|---|
| SPY 771.30, option bid/ask quotes, ^IRX 4.07% | Observed (Yahoo Finance via yfinance, 2026-09-25 16:00 ET) |
| Forward prices, implied vols, smile, term structure | Calculated here from the observed quotes |
| Price paths, hedging P&L | Simulated (GBM and jump-diffusion); fixed seeds |
| Transaction costs, grid, position size | Assumptions, listed below |

## 1. Pricing and implied vol

European Black–Scholes–Merton prices and Greeks (`src/options_lab/bsm.py`) and a bracketed implied-vol solver (`implied_vol.py`). Checked against a textbook example, put–call parity, no-arbitrage bounds, finite-difference Greeks and price→IV→price round trips (`tests/`, 75 tests).

At the 771 strike's implied vol (12.22%), the model prices the straddle at **$20.91 per share, $20,906 for the 10-lot** (market mid: $20.90). Position Greeks for the short 10-lot: vega −$1,692 per vol point, gamma −30 shares per $1 move, theta +$375 per day.

## 2. The real snapshot

Of 1,124 contracts across four expiries (7–84 days), 963 have usable quotes after flagging missing/zero bids, crossed markets, sub-5-cent prices and spreads wider than 50% of mid (`reports/snapshot_summary.md`).

**The forward comes from the quotes, not a dividend guess.** Put–call parity (C − P = D(F − K)) gives a forward for each strike; the median across near-the-money strikes varies by only $0.40–0.91 (IQR). For the Oct 23 expiry, F = 773.70.

**Yahoo's call/put IV gap is an artifact.** Yahoo reports call IVs about 2–4 vol points above put IVs at the same strike. With the parity-implied forward, our call and put IVs agree to a median of 0.17–0.37 points (`figures/03_call_put_iv_gap.png`). What gap remains is largest for in-the-money puts, as the American early-exercise premium would predict.

**Smile and term structure (one afternoon only).** Strong put skew: the 28-day IV is about 22% at 10% below the forward vs about 12% at the money. ATM vol rises with maturity: 11.0% (7d), 11.9% (28d), 13.3% (56d), 13.8% (84d) (`figures/03_iv_smile.png`, `03_term_structure.png`).

## 3. Hedging experiment (simulated)

**Design.** Sell the straddle at the model price at 12.22% vol. Simulate 20,000 paths over 28 days on a 260-step grid (13 half-hour decision points × 20 days). Compare, on **identical paths**:

1. no hedge;
2. rebalance to delta-neutral on a fixed schedule (every 30 min … every 5 days);
3. rebalance only when |net delta| exceeds a band (10 … 400 shares), checked every half hour.

Every policy uses BSM deltas at the implied vol and decides using only prices up to now. A self-financing cash ledger tracks premium, stock trades, interest, costs, liquidation and the final payoff. Base cost: $0.01 per share traded (sensitivity: $0 and $0.05). Which band counts as "matched in cost" to daily hedging was chosen on a separate set of development paths; every number below comes from the test paths. Full tables with Monte Carlo standard errors: `reports/hedging_results.md`.

### When the model is right (realized vol = implied vol)

| Policy ($0.01/share) | P&L std | Worst-5% average | Avg cost |
|---|---:|---:|---:|
| No hedge | $15,922 | −$40,876 | $0 |
| Daily schedule | $3,954 | −$9,379 | $37 |
| **Band 200 shares** (matched cost) | **$2,679** | **−$5,806** | $44 |
| Every 2 hours | $2,217 | −$5,110 | $61 |
| Band 100 shares | $1,667 | −$3,606 | $66 |
| Every 30 minutes | $1,122 | −$2,689 | $113 |

- **Daily hedging removes about 75% of the P&L standard deviation, for about $37 of costs on a $20,906 premium.**
- **A delta band beats a fixed schedule at similar cost:** at matched cost, the band had 32% lower std and a 38% smaller tail loss. It trades when the risk is there (after big moves) rather than on a clock. See `figures/04_risk_vs_cost.png`.
- For a product as liquid as SPY, costs are small next to the risk: even at $0.05/share, half-hourly hedging costs $563.
- Sanity check: with no costs, mean P&L is zero within its standard error, and the dispersion falls in proportion to √(rebalancing interval) (`figures/04_convergence.png`).

### When the model is wrong

| Scenario (every 30 min, $0.01/share) | Mean P&L | P&L std | Worst-5% average |
|---|---:|---:|---:|
| Realized = implied (12.2%) | −$106 | $1,122 | −$2,689 |
| Realized = 1.5 × implied | **−$10,515** | $5,075 | −$21,829 |
| Realized = 0.75 × implied | **+$5,084** | $1,859 | +$1,789 |
| Jumps, same total variance | +$53 | $5,949 | **−$21,005** |

- **Delta hedging removes direction risk, not volatility risk.** If realized vol comes in 50% above implied, the hedged position loses about $10.5k *whatever* the rebalancing policy. That's close to vega × the vol gap ($1,692 × 6.1 points ≈ $10.3k). A short straddle is a bet that realized vol stays below implied vol.
- **Jumps defeat delta hedging.** With the same total variance but some of it arriving as jumps, hedging every 30 minutes still leaves a worst-5% loss of about $21k (vs $2.7k without jumps), and rebalancing faster barely helps. You can't rebalance during a gap.

## Limitations

- Simulated paths from simple models (constant-vol GBM; Merton jumps with made-up parameters). Real markets have stochastic vol, vol that rises when prices fall, overnight gaps and weekends. The time grid is uniform.
- Hedging uses flat BSM deltas at one implied vol. It ignores the smile and how it moves (e.g. skew-adjusted delta).
- The options are treated as European and cash-settled at expiry. SPY options are American and physically settled.
- The option trade happens at the model mid. Selling at the bid would cost about $105 for the 10-lot (half the observed spreads); a market maker selling at the ask would earn it. Stock costs are linear per share, with no market impact.
- Drift = the risk-free rate (no view on direction). Results are for one strike, one expiry and one day's inputs.
- This is not a market-making simulation: no queue position, adverse selection, customer flow or spread capture.

## Reproduce

```bash
uv sync
uv run pytest                                   # 75 checks
uv run python scripts/analyze_snapshot.py       # needs a snapshot in data/raw/
uv run python scripts/run_hedging_experiment.py # ~30 s; no raw data needed
```
