# Gravitation Framework Backtester

An intraday confluence-scoring backtest engine (Central Pivot Range + Volume
Profile + intermarket correlation + PDH/PDL gap logic) built on Polygon.io
1-minute bars, plus an anchored walk-forward harness that re-tunes allocation
tiers out-of-sample.

## Contents

| File | Purpose |
|---|---|
| `Gravitation107.py` | Core strategy + single-pass backtest engine. Loads minute bars, computes CPR/VPOC/value-area, scores setups, sizes and manages trades, reports performance, plots equity curve. |
| `walk_forward.py` | Anchored walk-forward analysis on top of the strategy module. Grid-searches `ALLOCATION_TIERS` on an expanding train window each fold, freezes the winner, tests it on the next unseen window, and stitches the out-of-sample results together. |

## Important: this is a single in-sample backtest, made honest by the walk-forward harness

`Gravitation107.py` on its own reports results from **one static run over one
fixed date range**, with parameters (`ALLOCATION_TIERS`, `PTS_PDH_PDL_BREAK`,
`MAX_POSITIONS`, the `WATCHLIST`, etc.) that were hand-picked after looking at
that same period's results — see the comments in `CONFIG`. That makes the
headline numbers **in-sample / curve-fit by construction**, not a forecast of
future performance.

`walk_forward.py` exists specifically to pressure-test that number: it
re-optimizes the allocation-tier thresholds on expanding training windows and
evaluates only on data each fold has never seen, then reports the stitched
out-of-sample (OOS) Sharpe/return/drawdown alongside a Walk-Forward Efficiency
ratio (OOS Sharpe ÷ mean train Sharpe). A large gap between the in-sample and
OOS numbers is a strong signal of overfitting.

Note that the walk-forward harness only re-tunes `ALLOCATION_TIERS`. Other
hand-tuned choices baked into `Gravitation107.py`'s `CONFIG` (e.g. dropping
SPY from the watchlist, zeroing `PTS_PDH_PDL_BREAK`, `MAX_POSITIONS=3`) are
held fixed across every fold and are **not** validated out-of-sample by this
harness.

## Setup

```bash
git clone <this-repo-url>
cd <repo>
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Get a free or paid API key from [polygon.io](https://polygon.io) and set it
as an environment variable (do **not** hardcode it in the file):

```bash
export POLYGON_API_KEY="your_key_here"
```

See `.env.example` for the expected variable name.

## Usage

Run the single in-sample backtest first (this also populates the local
Parquet cache used by the walk-forward harness):

```bash
python Gravitation107.py
```

Outputs: `gravitation_backtest_results.csv`, `gravitation_equity_curve.png`,
`gravitation_backtest.log`.

Then run the walk-forward analysis (reuses the cache, makes no new API calls):

```bash
python walk_forward.py
```

Outputs: `walk_forward_folds.csv` (per-fold train/test params and metrics),
`walk_forward_oos_trades.csv` (stitched out-of-sample trade log).

Key knobs in `walk_forward.py`'s `WF` config: `TEST_DAYS` (fold length),
`MIN_TRAIN_DAYS` (size of the first anchored train window), `GRID_FLOOR` /
`GRID_MID` / `GRID_TOP` (allocation-tier breakpoints searched each fold), and
`OBJECTIVE` (`"sharpe"` or `"net_pnl"`).

## Notes / caveats

- Polygon's free tier is rate-limited (5 calls/min) and covers roughly two
  years of history; both scripts cache downloaded bars to `polygon_cache/`
  (Parquet) so re-runs are fast and don't re-hit the API.
- `walk_forward.py` monkey-patches `calc_volume_profile` with a vectorized
  NumPy version for speed; it's intended to be numerically identical to the
  original loop-based version, but hasn't been independently verified here —
  worth a spot-check against `Gravitation107.py`'s output on a few sessions
  before trusting OOS numbers for anything real.
- This is research/backtesting code, not a live trading system. Past
  (in-sample or out-of-sample) backtest performance is not a guarantee of
  future results.

## License
"All rights reserved — shared for portfolio/demonstration purposes only, not licensed for reuse or commercial use without permission."
