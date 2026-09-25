# Options Pricing and Hedging

A learning project that goes from market option quotes to risk management.

**Research question:** If I sell an option position, what risks do I take on, how can I measure them, and how much does it cost to reduce the directional exposure?

## Plan

1. **Payoffs and pricing.** Call and put payoffs, straddles, and European Black–Scholes–Merton prices.
2. **Greeks and implied volatility.** Delta, gamma, vega and theta, with numerical checks. Solving for implied volatility from a price.
3. **Real option-chain snapshot.** One timestamped SPY chain: quote quality, spreads, volatility smile and term structure.
4. **Hedging experiment.** A short at-the-money straddle on simulated price paths. It compares no hedge, hedging on a fixed schedule, and hedging when delta crosses a threshold, including transaction costs.

## Status

In progress. Nothing below this line counts as a result until it has been run and checked.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                       # create .venv with pinned dependencies
uv run python scripts/fetch_snapshot.py       # save an option-chain snapshot (run in US market hours)
uv run pytest                                 # run checks
```

## Data

Option quotes come from Yahoo Finance through the unofficial `yfinance` library, for personal research use. Raw data is not committed. Simulated data is always labeled as simulated.

## Limitations

This is an educational pricing and hedging study, not a market-making simulator. It does not model queue position, adverse selection, customer flow or achievable spread capture. SPY options are American-style and physically settled, so European pricing is only an approximation for them.
