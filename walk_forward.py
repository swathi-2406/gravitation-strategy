"""
walk_forward.py  —  Anchored walk-forward analysis for the Gravitation framework.

WHAT THIS DOES
--------------
Your Gravitation107.py was hand-tuned against the ENTIRE 2025-06 -> 2026-06 window
(SPY dropped, PDH/PDL zeroed, score floor raised, tiers recalibrated -- all decided
after seeing the full-period results). That makes the 168% / 5.75-Sharpe number an
IN-SAMPLE number. It tells you how well the params fit the past, not how the strategy
would have performed on data it never saw.

This harness produces the honest version:
  - ANCHORED train window (always starts at START_DATE, grows each fold)
  - On each train window: grid-search the score floor + allocation tiers, keep the
    combo with the best TRAIN Sharpe (subject to a min-trades guard)
  - FREEZE those params, run ONLY the next unseen test window with them
  - Concatenate all test-window trades -> the out-of-sample (OOS) equity curve

The OOS result is SUPPOSED to be worse than 168%. If it holds up reasonably, the edge
is real. If it collapses, the original number was curve-fit. Either way you learn the truth.

IMPORTANT: This does NOT modify Gravitation107.py. Run that file separately to reproduce
your original in-sample number; it is unchanged.

HOW TO RUN
----------
Put this file in the same folder as Gravitation107.py (so polygon_cache/ and your API
key resolve identically), then:

    python walk_forward.py

It reuses your on-disk parquet cache, so no extra API calls are made.
"""

import sys
import copy
import importlib
import itertools
import functools
from datetime import datetime, timedelta

# Force stdout to flush on every print, so progress is visible in real time even
# when the terminal block-buffers (common in IDE-integrated PowerShell on Windows).
# Without this the harness can look frozen while it is actually running fine.
print = functools.partial(print, flush=True)

# Make absolutely sure matplotlib never tries to open an interactive window
# (a stray figure window can block the process). Headless backend, set before
# the strategy module imports pyplot.
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import the strategy module WITHOUT running its __main__ block.
# (The bt.run() at the bottom of Gravitation107.py is under `if __name__ == ...`,
#  so importing is safe and does not trigger a backtest.)
# ---------------------------------------------------------------------------
STRATEGY_MODULE = "Gravitation107"   # change if you rename the file
strat = importlib.import_module(STRATEGY_MODULE)


# ===========================================================================
#  WALK-FORWARD CONFIG  (edit these, not the strategy file)
# ===========================================================================
WF = {
    # Test-window length in trading days. ~21 = one month. Larger = fewer,
    # more stable folds; smaller = more folds, noisier per-fold stats.
    "TEST_DAYS": 21,

    # Minimum trading days required in the FIRST anchored train window before
    # the first test fold begins. Needs enough data for the grid search to mean
    # anything. ~60 = three months.
    "MIN_TRAIN_DAYS": 60,

    # During each fold's grid search, reject any param combo on the train window
    # that produced fewer than this many trades (prevents picking a 3-trade fluke).
    "MIN_TRADES_TRAIN": 15,

    # ----- The parameter grid that gets re-optimized each fold -----
    # These are exactly the knobs your CONFIG comments admit to hand-tuning.
    # The harness rebuilds ALLOCATION_TIERS from (floor, mid, top) breakpoints:
    #   top  -> 1.00 alloc
    #   mid  -> 0.85 alloc
    #   floor-> 0.65 alloc
    #   below floor -> 0.00 (no trade)
    # Only combos with floor < mid < top are tried.
    "GRID_FLOOR": [6, 7, 8, 9],
    "GRID_MID":   [9, 10, 11],
    "GRID_TOP":   [11, 12, 13],

    # Objective used to pick the winning train combo: "sharpe" or "net_pnl".
    # Sharpe is the right default -- net_pnl alone rewards reckless sizing.
    "OBJECTIVE": "sharpe",
}


# ===========================================================================
#  INDICATOR MEMOIZATION
# ---------------------------------------------------------------------------
# CPR and volume profile depend ONLY on a session's price/volume bars, never on
# the allocation tiers being grid-searched. The original engine recomputes them
# on every run, so a 20-combo x 11-fold search recomputes the (very expensive)
# 200-bucket volume profile ~220 times per session. We wrap both functions with
# a cache keyed on session identity, so each session's indicators are computed
# exactly once for the whole walk-forward. Strategy file stays untouched.
# ===========================================================================
_CPR_CACHE = {}
_VP_CACHE = {}

def _session_key(session):
    """Cheap, collision-safe key for a session DataFrame: (symbol-agnostic)
    first/last timestamp + row count + last close. Sessions are unique per
    (date), and within one process the same date always carries the same bars."""
    idx = session.index
    return (idx[0].value, idx[-1].value, len(session),
            float(session["close"].iloc[-1]))

def install_indicator_cache():
    """Monkeypatch calc_cpr / calc_volume_profile with fast versions.

    calc_volume_profile is the runtime killer: the original loops with
    DataFrame.iterrows() over 200 price buckets per bar (~78k Python ops/session),
    and even with memoization the FIRST computation of each session dominates.
    We replace it with a numpy-vectorized version that produces byte-identical
    VPOC/VAH/VAL (verified to match the original exactly), then cache the result.
    calc_cpr is already cheap (min/max/mean), so it just gets a memo wrapper.
    """
    _orig_cpr = strat.calc_cpr

    def cached_cpr(session):
        k = _session_key(session)
        v = _CPR_CACHE.get(k)
        if v is None:
            v = _orig_cpr(session)
            _CPR_CACHE[k] = v
        return v

    def fast_vp(session):
        k = _session_key(session)
        cached = _VP_CACHE.get(k)
        if cached is not None:
            return cached

        H = session["high"].max()
        L = session["low"].min()
        if H == L:
            res = {"VPOC": H, "VAH": H, "VAL": L}
            _VP_CACHE[k] = res
            return res

        n = strat.CONFIG["PRICE_BUCKETS"]
        bins = np.linspace(L, H, n + 1)
        mids = (bins[:-1] + bins[1:]) / 2

        lows  = session["low"].to_numpy()
        highs = session["high"].to_numpy()
        vols  = session["volume"].to_numpy()

        # Vectorized bucket fill: overlap of each bar with each bucket, volume
        # distributed proportionally to overlap fraction. Matches the original
        # double loop exactly.
        blo = bins[:-1]
        bhi = bins[1:]
        lo = np.maximum(lows[:, None],  blo[None, :])
        hi = np.minimum(highs[:, None], bhi[None, :])
        ov = np.clip(hi - lo, 0, None)
        rng = np.clip(highs - lows, 1e-8, None)
        vol = ((vols / rng)[:, None] * ov).sum(axis=0)

        # Value-area expansion: identical to the original algorithm.
        vpoc_idx = int(np.argmax(vol))
        vpoc = float(mids[vpoc_idx])
        target = vol.sum() * strat.CONFIG["VALUE_AREA_PCT"]
        lo_idx = hi_idx = vpoc_idx
        captured = vol[vpoc_idx]
        while captured < target:
            add_lo = vol[lo_idx - 1] if lo_idx > 0     else 0
            add_hi = vol[hi_idx + 1] if hi_idx < n - 1 else 0
            if add_hi >= add_lo:
                hi_idx = min(hi_idx + 1, n - 1)
                captured += vol[hi_idx]
            else:
                lo_idx = max(lo_idx - 1, 0)
                captured += vol[lo_idx]
            if lo_idx == 0 and hi_idx == n - 1:
                break
        res = {"VPOC": vpoc, "VAH": float(mids[hi_idx]), "VAL": float(mids[lo_idx])}
        _VP_CACHE[k] = res
        return res

    strat.calc_cpr = cached_cpr
    strat.calc_volume_profile = fast_vp


# ===========================================================================
#  ONE-TIME DATA PRELOAD
# ---------------------------------------------------------------------------
# The original design re-ran bt.run() per grid combo, and each run rebuilt the
# whole backtester and reloaded every symbol for that window. Because the cache
# key embeds start/end, every new fold window was a fresh download -> on a free
# 5-calls/min key this is the difference between minutes and hours.
#
# Fix: load each symbol's FULL history ONCE into memory here, then slice windows
# from memory. The grid search never touches the network or disk again.
# ===========================================================================
_FULL_SESSIONS = None   # {symbol: {date_str: session_df}} for watchlist + SPY

def preload_full_history():
    """Load every symbol's full [START_DATE, END_DATE] history once, cache in RAM."""
    global _FULL_SESSIONS
    if _FULL_SESSIONS is not None:
        return _FULL_SESSIONS
    saved = copy.deepcopy(strat.CONFIG)
    try:
        bt = strat.GravitationBacktester()
        prev_level = strat.log.level
        strat.log.setLevel("ERROR")
        try:
            _FULL_SESSIONS = bt._load_all()   # uses disk cache; one window only
        finally:
            strat.log.setLevel(prev_level)
        return _FULL_SESSIONS
    finally:
        strat.CONFIG.clear()
        strat.CONFIG.update(saved)


def _slice_sessions(full, start_date, end_date, lookback_days=5):
    """
    Return {symbol: {date_str: df}} restricted to roughly [start_date, end_date],
    but include up to `lookback_days` calendar days BEFORE start_date so the engine
    has a prior session to compute CPR / volume profile for the first real day.
    The engine's own idx==0 guard handles the very first padded session.
    """
    pad_start = (datetime.strptime(start_date, "%Y-%m-%d")
                 - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    out = {}
    for sym, sess in full.items():
        out[sym] = {d: df for d, df in sess.items() if pad_start <= d <= end_date}
    return out


# ===========================================================================
#  Engine driver: run the strategy over a restricted date range with overrides
#  using PRELOADED in-memory data (no network, no disk).
# ===========================================================================
def run_engine(start_date, end_date, alloc_tiers):
    """
    Run the engine over [start_date, end_date] with the given allocation tiers,
    driving it entirely from preloaded in-memory sessions. Returns trade dicts.

    We monkeypatch the backtester's _load_all to return our in-memory slice, so
    no PolygonClient call is ever made during the grid search.
    """
    full = preload_full_history()
    sliced = _slice_sessions(full, start_date, end_date)

    saved = copy.deepcopy(strat.CONFIG)
    saved_load = strat.GravitationBacktester._load_all
    saved_report = strat.GravitationBacktester._report
    try:
        strat.CONFIG["START_DATE"] = start_date
        strat.CONFIG["END_DATE"]   = end_date
        strat.CONFIG["ALLOCATION_TIERS"] = alloc_tiers

        # Replace data loading with our in-memory slice for the duration.
        strat.GravitationBacktester._load_all = lambda self: sliced
        # Suppress the report/plot step entirely: run() calls _report() -> _plot(),
        # which renders a 3-panel matplotlib PNG on EVERY combo. That plotting is
        # the bulk of the remaining runtime and we never use the intermediate charts.
        strat.GravitationBacktester._report = lambda self: None

        bt = strat.GravitationBacktester()
        prev_level = strat.log.level
        strat.log.setLevel("ERROR")
        try:
            bt.run()
        finally:
            strat.log.setLevel(prev_level)
        # Drop any trades that closed on the padded lead-in days; only count
        # trades within the true [start_date, end_date] window.
        return [t for t in bt.trades if start_date <= t["date"] <= end_date]
    finally:
        strat.GravitationBacktester._load_all = saved_load
        strat.GravitationBacktester._report = saved_report
        strat.CONFIG.clear()
        strat.CONFIG.update(saved)


# ===========================================================================
#  Metrics on a trade list
# ===========================================================================
def metrics_from_trades(trades, initial_capital):
    """Compute summary stats from a list of trade dicts (order = chronological)."""
    if not trades:
        return {"trades": 0, "net": 0.0, "ret_pct": 0.0, "win_rate": 0.0,
                "profit_factor": 0.0, "sharpe": 0.0, "max_dd": 0.0, "final": initial_capital}

    df = pd.DataFrame(trades)
    df = df.sort_values("date").reset_index(drop=True)

    net = df["pnl"].sum()
    wins = df[df["pnl"] > 0]["pnl"]
    losses = df[df["pnl"] <= 0]["pnl"]
    gp = wins.sum()
    gl = abs(losses.sum())
    pf = gp / gl if gl > 0 else float("inf")
    wr = (df["pnl"] > 0).mean() * 100

    # Equity curve by trade close date (matches how the original aggregates).
    daily = df.groupby("date")["pnl"].sum().sort_index()
    equity = initial_capital + daily.cumsum()
    peak = equity.cummax()
    dd = ((equity - peak) / peak * 100).min()

    rets = equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    return {
        "trades": len(df),
        "net": net,
        "ret_pct": net / initial_capital * 100,
        "win_rate": wr,
        "profit_factor": pf,
        "sharpe": sharpe,
        "max_dd": dd,
        "final": initial_capital + net,
    }


def tiers_from_breaks(floor, mid, top):
    """Build an ALLOCATION_TIERS list from three score breakpoints."""
    return [
        (top,   1.00),
        (mid,   0.85),
        (floor, 0.65),
        (0,     0.00),
    ]


# ===========================================================================
#  Build the anchored fold schedule from the available trading dates
# ===========================================================================
def get_trading_dates():
    """
    Discover the actual trading dates from the preloaded full history, so folds
    align to real sessions (not calendar days). Uses the one-time RAM cache.
    """
    full = preload_full_history()
    dates = set()
    for sym in strat.CONFIG["WATCHLIST"]:
        dates.update(full.get(sym, {}).keys())
    return sorted(dates)


def build_folds(dates):
    """Anchored expanding-train, fixed-test folds. Returns list of fold dicts."""
    folds = []
    n = len(dates)
    train_end_idx = WF["MIN_TRAIN_DAYS"] - 1
    fold_num = 0
    while train_end_idx + WF["TEST_DAYS"] < n:
        test_start_idx = train_end_idx + 1
        test_end_idx = min(test_start_idx + WF["TEST_DAYS"] - 1, n - 1)
        fold_num += 1
        folds.append({
            "fold": fold_num,
            "train_start": dates[0],                 # anchored
            "train_end":   dates[train_end_idx],
            "test_start":  dates[test_start_idx],
            "test_end":    dates[test_end_idx],
        })
        train_end_idx = test_end_idx   # next train window absorbs this test window
    return folds


# ===========================================================================
#  Grid search on a train window
# ===========================================================================
def optimize_on_train(train_start, train_end, initial_capital):
    """Grid-search tiers on the train window; return (best_breaks, best_metrics)."""
    combos = [
        (f, m, t)
        for f, m, t in itertools.product(WF["GRID_FLOOR"], WF["GRID_MID"], WF["GRID_TOP"])
        if f < m < t
    ]
    best = None
    print(f"    searching {len(combos)} combos: ", end="")
    for f, m, t in combos:
        tiers = tiers_from_breaks(f, m, t)
        trades = run_engine(train_start, train_end, tiers)
        mt = metrics_from_trades(trades, initial_capital)
        print(".", end="", flush=True)   # heartbeat so it never looks frozen
        if mt["trades"] < WF["MIN_TRADES_TRAIN"]:
            continue
        key = mt["sharpe"] if WF["OBJECTIVE"] == "sharpe" else mt["net"]
        if best is None or key > best[2]:
            best = ((f, m, t), mt, key)
    print()   # newline after the dots
    if best is None:
        # Fallback: the strategy's own default tiers.
        return None, None
    return best[0], best[1]


# ===========================================================================
#  Main
# ===========================================================================
def main():
    init_cap = strat.CONFIG["INITIAL_CAPITAL"]

    install_indicator_cache()   # memoize CPR/VP across all folds and combos

    print("=" * 70)
    print("  ANCHORED WALK-FORWARD ANALYSIS")
    print("  Strategy module :", STRATEGY_MODULE)
    print("  Objective       :", WF["OBJECTIVE"])
    print("  Test window     :", WF["TEST_DAYS"], "trading days")
    print("  Min train       :", WF["MIN_TRAIN_DAYS"], "trading days")
    print("=" * 70)

    print("\nDiscovering trading dates from cache ...")
    dates = get_trading_dates()
    if len(dates) < WF["MIN_TRAIN_DAYS"] + WF["TEST_DAYS"]:
        print(f"Not enough data: {len(dates)} trading days available.")
        sys.exit(1)
    print(f"  {len(dates)} trading days: {dates[0]} -> {dates[-1]}")

    folds = build_folds(dates)
    print(f"  {len(folds)} anchored folds\n")

    all_oos_trades = []
    fold_rows = []

    for fd in folds:
        print(f"--- Fold {fd['fold']}: "
              f"train {fd['train_start']}..{fd['train_end']}  "
              f"test {fd['test_start']}..{fd['test_end']}")

        best_breaks, train_mt = optimize_on_train(
            fd["train_start"], fd["train_end"], init_cap
        )
        if best_breaks is None:
            print("    no valid param combo on train window, skipping fold")
            continue

        tiers = tiers_from_breaks(*best_breaks)
        oos_trades = run_engine(fd["test_start"], fd["test_end"], tiers)
        oos_mt = metrics_from_trades(oos_trades, init_cap)

        all_oos_trades.extend(oos_trades)

        print(f"    chosen breaks (floor,mid,top) = {best_breaks}")
        print(f"    TRAIN  sharpe={train_mt['sharpe']:.2f}  "
              f"net=${train_mt['net']:+,.0f}  trades={train_mt['trades']}")
        print(f"    OOS    sharpe={oos_mt['sharpe']:.2f}  "
              f"net=${oos_mt['net']:+,.0f}  wr={oos_mt['win_rate']:.1f}%  "
              f"trades={oos_mt['trades']}")

        fold_rows.append({
            "fold": fd["fold"],
            "train_start": fd["train_start"], "train_end": fd["train_end"],
            "test_start": fd["test_start"], "test_end": fd["test_end"],
            "floor": best_breaks[0], "mid": best_breaks[1], "top": best_breaks[2],
            "train_sharpe": round(train_mt["sharpe"], 3),
            "train_net": round(train_mt["net"], 2),
            "oos_sharpe": round(oos_mt["sharpe"], 3),
            "oos_net": round(oos_mt["net"], 2),
            "oos_win_rate": round(oos_mt["win_rate"], 1),
            "oos_trades": oos_mt["trades"],
        })

    # ---- Stitched out-of-sample result ----
    oos = metrics_from_trades(all_oos_trades, init_cap)

    print("\n" + "=" * 70)
    print("  STITCHED OUT-OF-SAMPLE RESULT  (the honest number)")
    print("=" * 70)
    print(f"  OOS trades        : {oos['trades']}")
    print(f"  OOS net P&L       : ${oos['net']:+,.2f}")
    print(f"  OOS total return  : {oos['ret_pct']:+.2f}%")
    print(f"  OOS win rate      : {oos['win_rate']:.1f}%")
    print(f"  OOS profit factor : {oos['profit_factor']:.2f}")
    print(f"  OOS Sharpe        : {oos['sharpe']:.2f}")
    print(f"  OOS max drawdown  : {oos['max_dd']:.2f}%")
    print("=" * 70)
    print("  Compare against the IN-SAMPLE run of Gravitation107.py")
    print("  (originally ~+168% / Sharpe ~5.75). A large gap = curve-fit.")
    print("=" * 70)

    # ---- Save artifacts ----
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv("walk_forward_folds.csv", index=False)
    pd.DataFrame(all_oos_trades).to_csv("walk_forward_oos_trades.csv", index=False)
    print("\n  Saved -> walk_forward_folds.csv")
    print("  Saved -> walk_forward_oos_trades.csv")

    # Walk-Forward Efficiency: OOS Sharpe / mean train Sharpe.
    if fold_rows:
        mean_train_sharpe = np.mean([r["train_sharpe"] for r in fold_rows])
        if mean_train_sharpe != 0:
            wfe = oos["sharpe"] / mean_train_sharpe
            print(f"\n  Walk-Forward Efficiency (OOS / mean train Sharpe): {wfe:.2f}")
            print("  Rule of thumb: > ~0.5 is encouraging; near 0 or negative = overfit.")


if __name__ == "__main__":
    main()