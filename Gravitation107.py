import os
import sys
import io
import math
import time
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")





CONFIG = {
    # -- Polygon.io API key --------------------------------------------------
    "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY", ""),

    # -- Backtest window -----------------------------------------------------
    # Polygon free tier covers ~2 years of history.
    "START_DATE": "2025-06-30",
    "END_DATE":   "2026-06-27",

    # -- Universe ------------------------------------------------------------
    # SPY removed: consistently losing (-$4,473, 35.9% WR) because it's the
    # most arbitraged ETF. Informational edge lives in NVDA, INTC, QQQ, XLK.
    # SPY is still loaded as CORR_ASSET for intermarket correlation checks.
    "WATCHLIST": ["X:BTCUSD", "X:ETHUSD", "X:SOLUSD"],

    # -- Starting capital ----------------------------------------------------
    "INITIAL_CAPITAL": 100_000.0,

    # -- Strategy parameters -------------------------------------------------
    "VALUE_AREA_PCT":     0.40,
    "PRICE_BUCKETS":      200,
    "CPR_NARROW_PCT":     0.0025,
    "CPR_WIDE_PCT":       0.0050,

    # Confluence weights
    "PTS_BREAKOUT":       3,
    "PTS_VPOC_RETEST":    3,
    "PTS_NARROW_CPR":     2,
    "PTS_INTERMARKET":    2,
    "PTS_ADV_CHANGE":     1,
    "PTS_NORMAL_CPR":     1,

    # --- New Gravitation filters (added) -----------------------------------
    # CPR directional confirmation: keeps contributing to scoring
    "PTS_ABOVE_CPR":      1,
    # POC directional confirmation: keeps contributing to scoring
    "PTS_ABOVE_POC":      1,
    # PDH/PDL break weight set to 0.
    # Evidence: scores 14-16 (reached only via PDH/PDL break) had 25-29% win
    # rates — WORSE than scores 8-13. The confirmed-break is a LATE entry
    # signal that inflated scores without improving edge. Zeroing it moves
    # those same trades back to scores 11-13 where performance is strongest.
    "PTS_PDH_PDL_BREAK":  0,
    "PTS_GAP_CONT":       0,

    # PDH/PDL gate remains off (soft, score-only). Already confirmed off.
    "REQUIRE_PDH_PDL_BREAK": False,

    # Allocation tiers: recalibrated for the new score ceiling of 13.
    # (Without PDH/PDL weights: max = 3+3+2+2+1+1+1 = 13)
    # Score floor raised to 8: scores 6-7 are weak setups with negative or
    # zero win rates and not enough trades to trust.
    "ALLOCATION_TIERS": [
        (12, 1.00),
        (10, 0.85),
        ( 8, 0.65),
        ( 0, 0.00),   # scores below 8 do not trade
    ],

    # Risk management
    "MAX_RISK_PER_TRADE":  0.08,
    "MIN_RR_RATIO":        2.5,
    "STOP_BUFFER_PCT":     0.002,
    "TRAILING_BUFFER_PCT": 0.007,
    "VPOC_TOUCH_TOL_PCT":  0.003,

    # Optional: cap capital deployed per trade as % of equity (None = no cap)
    "MAX_POSITION_PCT":    0.50,

    # Execution timing
    "OPENING_FILTER_MINS": 15,
    "SESSION_CLOSE_MINS":  15,    # force-close N mins before 16:00

    # Max simultaneous positions raised to 3.
    # With 4 symbols (SPY removed) and max drawdown only at 2.96%, there is
    # genuine headroom to capture a third concurrent signal.
    # Avg loss per trade ~$324; 3 simultaneous stops ≈ 0.6-1% of equity.
    "MAX_POSITIONS": 3,

    # Slippage per side (0.05%)
    "SLIPPAGE_PCT": 0.0005,

    # Intermarket asset
    "CORR_ASSET": "SPY",

    # Polygon rate limiting (free tier = 5 calls/min)
    "API_CALLS_PER_MIN": 5,

    # Cache downloaded data to disk so re-runs are instant
    "USE_CACHE": True,
    "CACHE_DIR": "polygon_cache",
}

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler("gravitation_backtest.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("GravBT")


# -----------------------------------------------------------------------------
#  POLYGON DATA CLIENT
# -----------------------------------------------------------------------------

class PolygonClient:
    """Fetches 1-minute aggregate bars from Polygon.io with rate limiting + caching."""

    BASE = "https://api.polygon.io"

    def __init__(self, api_key: str):
        self.api_key   = api_key
        self.last_call = 0.0
        self.min_gap   = 60.0 / CONFIG["API_CALLS_PER_MIN"]   # seconds between calls
        if CONFIG["USE_CACHE"]:
            os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)

    def _throttle(self):
        """Respect the free-tier rate limit."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_gap:
            time.sleep(self.min_gap - elapsed)
        self.last_call = time.time()

    def _cache_path(self, symbol: str, start: str, end: str) -> str:
        return os.path.join(CONFIG["CACHE_DIR"], f"{symbol}_{start}_{end}_1min.parquet")

    def get_minute_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        Fetch 1-minute bars for [start, end] inclusive.
        Polygon paginates results; we follow next_url until done.
        Returns a DataFrame indexed by ET timestamp with OHLCV columns.
        """
        # Check cache first
        if CONFIG["USE_CACHE"]:
            cpath = self._cache_path(symbol, start, end)
            if os.path.exists(cpath):
                try:
                    df = pd.read_parquet(cpath)
                    log.info(f"  {symbol}: loaded from cache ({len(df)} bars)")
                    return df
                except Exception:
                    pass  # corrupt cache, re-download

        url = (f"{self.BASE}/v2/aggs/ticker/{symbol}/range/1/minute/"
               f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")

        all_results = []
        page = 0
        while url:
            self._throttle()
            try:
                resp = requests.get(url, timeout=30)
            except Exception as e:
                log.error(f"  {symbol}: request failed - {e}")
                break

            if resp.status_code == 429:
                log.warning(f"  {symbol}: rate limited, waiting 60s ...")
                time.sleep(60)
                continue
            if resp.status_code == 401:
                log.error(f"  {symbol}: 401 Unauthorized - check your API key")
                break
            if resp.status_code != 200:
                log.error(f"  {symbol}: HTTP {resp.status_code} - {resp.text[:200]}")
                break

            data = resp.json()
            results = data.get("results", [])
            all_results.extend(results)
            page += 1

            # Polygon returns next_url for pagination
            next_url = data.get("next_url")
            if next_url:
                url = f"{next_url}&apiKey={self.api_key}"
            else:
                url = None

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        # Polygon columns: t=timestamp(ms), o,h,l,c=OHLC, v=volume, vw=vwap, n=trades
        df = df.rename(columns={
            "o": "open", "h": "high", "l": "low",
            "c": "close", "v": "volume", "t": "timestamp",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_convert(ET)
        df = df[["open", "high", "low", "close", "volume"]]

        # Keep only Regular Trading Hours (09:30 - 16:00 ET)
        df = df.between_time("09:30", "16:00")

        # Save to cache
        if CONFIG["USE_CACHE"] and not df.empty:
            try:
                df.to_parquet(self._cache_path(symbol, start, end))
            except Exception as e:
                log.warning(f"  {symbol}: could not cache - {e}")

        return df


def split_sessions(df: pd.DataFrame) -> dict:
    """Split continuous minute bars into {date_str: session_df}."""
    sessions = {}
    if df.empty:
        return sessions
    for date, group in df.groupby(df.index.date):
        if not group.empty:
            sessions[str(date)] = group
    return sessions


# -----------------------------------------------------------------------------
#  STRATEGY CALCULATIONS  (operate on 1-min session bars)
# -----------------------------------------------------------------------------

def calc_cpr(session: pd.DataFrame) -> dict:
    """Central Pivot Range from a full session of 1-min bars."""
    H = session["high"].max()
    L = session["low"].min()
    C = float(session["close"].iloc[-1])
    P  = (H + L + C) / 3
    BC = (H + L) / 2
    TC = (P - BC) + P
    if TC < BC:
        TC, BC = BC, TC
    width_pct = (TC - BC) / P if P else 0
    if width_pct < CONFIG["CPR_NARROW_PCT"]:
        day_type = "TREND"
    elif width_pct > CONFIG["CPR_WIDE_PCT"]:
        day_type = "ROTATIONAL"
    else:
        day_type = "NORMAL"
    return {"P": P, "BC": BC, "TC": TC, "width_pct": width_pct, "day_type": day_type}


def calc_volume_profile(session: pd.DataFrame) -> dict:
    """Spatial volume profile with 40% value area from 1-min bars."""
    H = session["high"].max()
    L = session["low"].min()
    if H == L:
        return {"VPOC": H, "VAH": H, "VAL": L}
    n    = CONFIG["PRICE_BUCKETS"]
    bins = np.linspace(L, H, n + 1)
    mids = (bins[:-1] + bins[1:]) / 2
    vol  = np.zeros(n)
    for _, row in session.iterrows():
        bar_lo, bar_hi, v = row["low"], row["high"], row["volume"]
        rng = bar_hi - bar_lo or 1e-8
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            overlap = max(0, min(bar_hi, hi) - max(bar_lo, lo))
            vol[i] += v * (overlap / rng)
    vpoc_idx = int(np.argmax(vol))
    vpoc     = float(mids[vpoc_idx])
    target   = vol.sum() * CONFIG["VALUE_AREA_PCT"]
    lo_idx = hi_idx = vpoc_idx
    captured = vol[vpoc_idx]
    while captured < target:
        add_lo = vol[lo_idx - 1] if lo_idx > 0     else 0
        add_hi = vol[hi_idx + 1] if hi_idx < n - 1 else 0
        if add_hi >= add_lo:
            hi_idx   = min(hi_idx + 1, n - 1)
            captured += vol[hi_idx]
        else:
            lo_idx   = max(lo_idx - 1, 0)
            captured += vol[lo_idx]
        if lo_idx == 0 and hi_idx == n - 1:
            break
    return {"VPOC": vpoc, "VAH": float(mids[hi_idx]), "VAL": float(mids[lo_idx])}


def calc_prev_day_levels(prev_bars: pd.DataFrame) -> dict:
    """Previous Day High / Low from the prior session's 1-min bars."""
    return {
        "PDH": float(prev_bars["high"].max()),
        "PDL": float(prev_bars["low"].min()),
    }


def detect_gap(today_bars: pd.DataFrame, pdh: float, pdl: float) -> str:
    """
    Classify the session open relative to the previous day's range.
      'GAP_UP'   -> opens above PDH
      'GAP_DOWN' -> opens below PDL
      'NONE'     -> opens inside the prior range
    Uses the first bar's open of the session.
    """
    if today_bars.empty:
        return "NONE"
    open_px = float(today_bars["open"].iloc[0])
    if open_px > pdh:
        return "GAP_UP"
    if open_px < pdl:
        return "GAP_DOWN"
    return "NONE"


def confirmed_break_above(bars: pd.DataFrame, level: float) -> bool:
    """
    Confirmed breakout above `level`: requires a candle CLOSE above the level,
    not just a wick/touch. Accepts a single confirmed close (the most recent bar)
    or two consecutive closes above the level. Returns False on an empty frame.
    """
    if bars is None or bars.empty:
        return False
    closes = bars["close"]
    last_close = float(closes.iloc[-1])
    if last_close <= level:
        return False
    # Single confirmed close above is sufficient; two consecutive closes also pass.
    return True


def confirmed_break_below(bars: pd.DataFrame, level: float) -> bool:
    """
    Confirmed breakdown below `level`: requires a candle CLOSE below the level,
    not just a wick/touch. Mirror of confirmed_break_above.
    """
    if bars is None or bars.empty:
        return False
    closes = bars["close"]
    last_close = float(closes.iloc[-1])
    if last_close >= level:
        return False
    return True


def gap_continuation_ok(side: str, bars: pd.DataFrame, gap_type: str,
                        pdh: float, pdl: float) -> bool:
    """
    Validates gap-day continuation / failed-gap logic.

    Gap-Up + long: market must still be HOLDING above PDH (current close above PDH);
    if price has fallen back below PDH it's a failed gap -> reject until reclaimed.

    Gap-Down + short: market must still be HOLDING below PDL (current close below PDL);
    if price has reclaimed PDL it's a failed gap -> reject until it breaks down again.

    For non-gap sessions this returns True (no gap constraint to apply).
    """
    if bars is None or bars.empty:
        return True
    last_close = float(bars["close"].iloc[-1])
    if side == "long" and gap_type == "GAP_UP":
        # Holding above PDH = valid continuation; below = failed gap.
        return last_close > pdh
    if side == "short" and gap_type == "GAP_DOWN":
        return last_close < pdl
    return True


def get_bias(price: float, vp: dict, cpr: dict) -> str:
    if price > vp["VAH"] and price > cpr["TC"]:
        return "BULLISH"
    if price < vp["VAL"] and price < cpr["BC"]:
        return "BEARISH"
    return "NEUTRAL"


def detect_vpoc_bounce(bars: pd.DataFrame, vpoc: float, bias: str) -> bool:
    """True if price touched near VPOC in last 3 candles and closed in bias direction."""
    if len(bars) < 3:
        return False
    w   = bars.iloc[-3:]
    tol = vpoc * CONFIG["VPOC_TOUCH_TOL_PCT"]
    touched = (
        ((w["low"]  <= vpoc + tol) & (w["low"]  >= vpoc - tol)).any() or
        ((w["high"] >= vpoc - tol) & (w["high"] <= vpoc + tol)).any() or
        ((w["low"] <= vpoc) & (w["high"] >= vpoc)).any()
    )
    if not touched:
        return False
    last = w.iloc[-1]
    if bias == "BULLISH":
        return last["close"] > last["open"]
    if bias == "BEARISH":
        return last["close"] < last["open"]
    return False


def score_setup(bias: str, cpr: dict, vol_ok: bool, corr: bool, vpoc_bounce: bool,
                above_cpr: bool = False, above_poc: bool = False,
                pdh_pdl_break: bool = False, gap_cont: bool = False) -> int:
    if bias == "NEUTRAL":
        return 0
    s = CONFIG["PTS_BREAKOUT"]            # bias locked = 3 pts
    if vpoc_bounce:
        s += CONFIG["PTS_VPOC_RETEST"]
    if cpr["day_type"] == "TREND":
        s += CONFIG["PTS_NARROW_CPR"]
    elif cpr["day_type"] == "NORMAL":
        s += CONFIG["PTS_NORMAL_CPR"]
    if corr:
        s += CONFIG["PTS_INTERMARKET"]
    if vol_ok:
        s += CONFIG["PTS_ADV_CHANGE"]
    # --- New Gravitation confluence filters ---
    if above_cpr:
        s += CONFIG["PTS_ABOVE_CPR"]
    if above_poc:
        s += CONFIG["PTS_ABOVE_POC"]
    if pdh_pdl_break:
        s += CONFIG["PTS_PDH_PDL_BREAK"]
    if gap_cont:
        s += CONFIG["PTS_GAP_CONT"]
    return s


def allocation_frac(score: int) -> float:
    for min_s, frac in CONFIG["ALLOCATION_TIERS"]:
        if score >= min_s:
            return frac
    return 0.0


def apply_slippage(price: float, side: str) -> float:
    slip = price * CONFIG["SLIPPAGE_PCT"]
    return price + slip if side == "long" else price - slip


def apply_exit_slippage(price: float, position_side: str) -> float:
    """
    Adverse slippage on exit. Closing a long = sell (fill lower);
    closing a short = buy-to-cover (fill higher). Always moves against us.
    """
    slip = price * CONFIG["SLIPPAGE_PCT"]
    return price - slip if position_side == "long" else price + slip


# -----------------------------------------------------------------------------
#  OPEN POSITION  (trailing stop logic)
# -----------------------------------------------------------------------------

class OpenPosition:
    def __init__(self, symbol, side, entry, stop, qty, score):
        self.symbol       = symbol
        self.side         = side
        self.entry        = entry
        self.stop         = stop
        self.initial_stop = stop   # preserved as the structural VPOC/ATR anchor
        self.qty          = qty
        self.score        = score
        self.breakeven    = False

    def update(self, high: float, low: float, price: float) -> bool:
        """
        Two-tier stop management:

        Tier 1 — Structural stop (initial_stop, VPOC-anchored):
          Checked INTRABAR using the bar's high/low. If the VPOC level is
          breached mid-bar the trade genuinely failed; we exit realistically.

        Tier 2 — Trailing/breakeven stop (stop after it moves beyond initial):
          Checked on the bar CLOSE only. The trailing buffer is calibrated for
          smooth close-to-close movement, not intrabar wicks. Checking intrabar
          against a 0.7% buffer on a 1-min NVDA bar is pure noise; close-only
          prevents that whipsaw while still managing the trade.

        `price` is the bar close; used for breakeven/trailing updates.
        """
        # --- Tier 1: structural stop — intrabar ---
        if self.side == "long"  and low  <= self.initial_stop:
            self.stop = self.initial_stop   # fill at structural level
            return True
        if self.side == "short" and high >= self.initial_stop:
            self.stop = self.initial_stop
            return True

        # --- Tier 2: trailed/breakeven stop — close only ---
        if self.stop != self.initial_stop:
            if self.side == "long"  and price <= self.stop:
                return True
            if self.side == "short" and price >= self.stop:
                return True

        # --- Update: breakeven shift then trail ---
        risk = abs(self.entry - self.initial_stop)
        buf  = price * CONFIG["TRAILING_BUFFER_PCT"]

        if not self.breakeven:
            if self.side == "long"  and price >= self.entry + risk:
                self.stop      = self.entry
                self.breakeven = True
            elif self.side == "short" and price <= self.entry - risk:
                self.stop      = self.entry
                self.breakeven = True

        if self.side == "long":
            self.stop = max(self.stop, price - buf)
        else:
            self.stop = min(self.stop, price + buf)
        return False


# -----------------------------------------------------------------------------
#  BACKTESTER ENGINE
# -----------------------------------------------------------------------------

class GravitationBacktester:

    def __init__(self):
        self.client       = PolygonClient(CONFIG["POLYGON_API_KEY"])
        self.equity       = CONFIG["INITIAL_CAPITAL"]
        self.trades       = []
        self.equity_curve = []

    # -- Data loading --------------------------------------------------------

    def _load_all(self) -> dict:
        """Returns {symbol: {date_str: session_df}} for all tickers + SPY."""
        symbols = list(dict.fromkeys(CONFIG["WATCHLIST"] + [CONFIG["CORR_ASSET"]]))
        all_sessions = {}
        for sym in symbols:
            log.info(f"  Loading {sym} ...")
            df = self.client.get_minute_bars(sym, CONFIG["START_DATE"], CONFIG["END_DATE"])
            if df.empty:
                log.warning(f"  {sym}: no data returned")
                all_sessions[sym] = {}
            else:
                all_sessions[sym] = split_sessions(df)
                log.info(f"  {sym}: {len(all_sessions[sym])} sessions, {len(df)} bars")
        return all_sessions

    # -- Main backtest loop --------------------------------------------------

    def run(self):
        log.info("=" * 62)
        log.info("  GRAVITATION FRAMEWORK BACKTESTER  (Polygon.io)")
        log.info(f"  Period : {CONFIG['START_DATE']} -> {CONFIG['END_DATE']}")
        log.info(f"  Capital: ${CONFIG['INITIAL_CAPITAL']:,.0f}")
        log.info(f"  Tickers: {', '.join(CONFIG['WATCHLIST'])}")
        log.info("=" * 62)

        if CONFIG["POLYGON_API_KEY"] in ("", "YOUR_POLYGON_API_KEY_HERE"):
            log.error("  No Polygon API key set! Edit CONFIG['POLYGON_API_KEY'] "
                      "or set the POLYGON_API_KEY environment variable.")
            return

        all_sessions = self._load_all()
        spy_sessions = all_sessions.get(CONFIG["CORR_ASSET"], {})

        # Build sorted list of trading dates in the window
        all_dates = set()
        for sym in CONFIG["WATCHLIST"]:
            all_dates.update(all_sessions.get(sym, {}).keys())
        sorted_dates = sorted(all_dates)
        log.info(f"  Trading days in window: {len(sorted_dates)}")

        if not sorted_dates:
            log.warning("  No data loaded - check your API key and date range.")
            return

        open_positions = {}   # symbol -> OpenPosition

        for date_str in sorted_dates:
            log.info(f"  -- {date_str}  equity=${self.equity:,.2f}  "
                     f"open={list(open_positions.keys())}")

            for sym in CONFIG["WATCHLIST"]:
                sym_sessions = all_sessions.get(sym, {})
                today_bars   = sym_sessions.get(date_str)
                if today_bars is None or today_bars.empty:
                    continue

                # Need previous session for CPR + VP
                sym_dates = sorted(sym_sessions.keys())
                try:
                    idx = sym_dates.index(date_str)
                except ValueError:
                    continue
                if idx == 0:
                    continue
                prev_bars = sym_sessions[sym_dates[idx - 1]]

                self._simulate_session(
                    date_str, sym, today_bars, prev_bars,
                    sym_sessions, sym_dates, idx,
                    spy_sessions, open_positions,
                )

            self.equity_curve.append({"date": date_str, "equity": round(self.equity, 2)})

        # Close any residual positions at final close
        for sym, pos in list(open_positions.items()):
            last_date = sorted_dates[-1]
            last_bars = all_sessions[sym].get(last_date)
            if last_bars is not None and not last_bars.empty:
                price = float(last_bars["close"].iloc[-1])
                self._close(open_positions, sym, price, last_date, "BT_END",
                            calc_cpr(last_bars)["day_type"])

        self._report()

    # -- Single session replay -----------------------------------------------

    def _simulate_session(self, date_str, symbol, today_bars, prev_bars,
                          sym_sessions, sym_dates, day_idx,
                          spy_sessions, open_positions):
        """Replay one RTH session bar-by-bar for one symbol."""
        try:
            cpr = calc_cpr(prev_bars)
            vp  = calc_volume_profile(prev_bars)
        except Exception as e:
            return

        # --- Previous Day High/Low + gap classification (added) ---
        try:
            pdlevels = calc_prev_day_levels(prev_bars)
            pdh = pdlevels["PDH"]
            pdl = pdlevels["PDL"]
        except Exception:
            return
        gap_type = detect_gap(today_bars, pdh, pdl)

        # 20-day average volume for ADV check
        hist_dates = sym_dates[max(0, day_idx - 20): day_idx]
        avg_vol = np.mean([sym_sessions[d]["volume"].sum() for d in hist_dates]) \
                  if hist_dates else 0

        spy_today = spy_sessions.get(date_str, pd.DataFrame())

        market_open  = today_bars.index[0].replace(hour=9,  minute=30, second=0, microsecond=0)
        close_cutoff = today_bars.index[0].replace(hour=16, minute=0,  second=0, microsecond=0) \
                       - timedelta(minutes=CONFIG["SESSION_CLOSE_MINS"])

        bias_locked  = False
        daily_bias   = None
        traded_today = False   # blocks re-entry after a stop-out same day

        for i in range(len(today_bars)):
            ts    = today_bars.index[i]
            bar   = today_bars.iloc[i]
            price = float(bar["close"])
            mins  = (ts - market_open).total_seconds() / 60

            # --- Manage existing position ---
            if symbol in open_positions:
                pos     = open_positions[symbol]
                bar_hi  = float(bar["high"])
                bar_lo  = float(bar["low"])
                stopped = pos.update(bar_hi, bar_lo, price)
                force   = ts >= close_cutoff
                if stopped or force:
                    if stopped:
                        exit_px = apply_exit_slippage(pos.stop, pos.side)
                    else:
                        exit_px = apply_exit_slippage(price, pos.side)
                    self._close(open_positions, symbol, exit_px, date_str,
                                "STOP" if stopped else "EOD", cpr["day_type"])
                    traded_today = True   # no re-entry this session after stop
                continue

            # --- Already traded today (entry or stop-out) ---
            if traded_today:
                continue

            # --- Opening filter ---
            if mins < CONFIG["OPENING_FILTER_MINS"]:
                continue

            # --- Lock bias once, right after the filter ---
            if not bias_locked:
                bias_locked = True
                daily_bias  = get_bias(price, vp, cpr)
                if daily_bias == "NEUTRAL":
                    return   # rotational day, sit out

            if daily_bias == "NEUTRAL":
                return

            # --- Max positions guard ---
            if len(open_positions) >= CONFIG["MAX_POSITIONS"]:
                continue

            # --- VPOC retest + bounce ---
            bars_so_far = today_bars.iloc[: i + 1]
            if not detect_vpoc_bounce(bars_so_far, vp["VPOC"], daily_bias):
                continue

            # --- Intermarket correlation (SPY up to this timestamp) ---
            if not spy_today.empty:
                spy_so_far = spy_today[spy_today.index <= ts]
                if len(spy_so_far) >= 2:
                    spy_bull = spy_so_far["close"].iloc[-1] > spy_so_far["open"].iloc[0]
                    corr = (daily_bias == "BULLISH" and spy_bull) or \
                           (daily_bias == "BEARISH" and not spy_bull)
                else:
                    corr = False
            else:
                corr = False

            # --- Volume vs 20-day ADV (projected) ---
            proj_vol = bars_so_far["volume"].sum() * (390 / max(mins, 1))
            vol_ok   = proj_vol > avg_vol if avg_vol else False

            # --- New Gravitation directional gates (added) -----------------
            # Trade side derived from the locked daily bias.
            side = "long" if daily_bias == "BULLISH" else "short"

            # CPR confluence: longs only above CPR (price > TC),
            # shorts only below CPR (price < BC).
            if side == "long":
                above_cpr = price > cpr["TC"]
            else:
                above_cpr = price < cpr["BC"]

            # POC confluence: longs only above POC, shorts only below POC.
            if side == "long":
                above_poc = price > vp["VPOC"]
            else:
                above_poc = price < vp["VPOC"]

            # Previous Day High/Low confirmed break (close beyond level, not a wick).
            if side == "long":
                pdh_pdl_break = confirmed_break_above(bars_so_far, pdh)
            else:
                pdh_pdl_break = confirmed_break_below(bars_so_far, pdl)

            # Gap-day continuation / failed-gap handling.
            gap_ok   = gap_continuation_ok(side, bars_so_far, gap_type, pdh, pdl)
            gap_cont = gap_ok and (
                (side == "long"  and gap_type == "GAP_UP") or
                (side == "short" and gap_type == "GAP_DOWN")
            )

            # Hard requirement gate: CPR + POC + confirmed PDH/PDL break must all
            # pass before an entry is allowed. Gap days must also not be a failed gap.
            if CONFIG["REQUIRE_PDH_PDL_BREAK"]:
                if not (above_cpr and above_poc and pdh_pdl_break and gap_ok):
                    continue

            # --- Confluence score ---
            score = score_setup(daily_bias, cpr, vol_ok, corr, True,
                                 above_cpr=above_cpr, above_poc=above_poc,
                                 pdh_pdl_break=pdh_pdl_break, gap_cont=gap_cont)
            frac  = allocation_frac(score)
            if frac == 0.0:
                continue

            # --- Entry, stop, target ---
            entry = apply_slippage(price, side)
            buf   = vp["VPOC"] * CONFIG["STOP_BUFFER_PCT"]
            # Structural stop at VPOC, or 1% ATR stop if VPOC too far
            vpoc_stop = (vp["VPOC"] - buf) if side == "long" else (vp["VPOC"] + buf)
            if abs(entry - vpoc_stop) <= entry * 0.02:
                stop = vpoc_stop
            else:
                stop = (entry * 0.99) if side == "long" else (entry * 1.01)

            risk = abs(entry - stop)
            if risk < 0.01:
                continue
            target = (entry + risk * CONFIG["MIN_RR_RATIO"]) if side == "long" \
                     else (entry - risk * CONFIG["MIN_RR_RATIO"])

            # Validate stop not already blown
            if side == "long"  and price <= stop:
                continue
            if side == "short" and price >= stop:
                continue

            # --- Position sizing ---
            dollar_risk = self.equity * CONFIG["MAX_RISK_PER_TRADE"] * frac
            qty = math.floor(dollar_risk / risk)

            # Optional cap on capital deployed
            if CONFIG["MAX_POSITION_PCT"]:
                max_notional = self.equity * CONFIG["MAX_POSITION_PCT"]
                max_qty      = math.floor(max_notional / entry)
                qty          = min(qty, max_qty)

            if qty <= 0:
                continue

            open_positions[symbol] = OpenPosition(symbol, side, entry, stop, qty, score)
            log.info(f"     ENTER {side.upper()} {qty} {symbol} @ {entry:.2f}  "
                     f"stop={stop:.2f}  target={target:.2f}  score={score}  "
                     f"[{date_str} {ts.strftime('%H:%M')}]")
            traded_today = True
            break   # one entry per symbol per day

    # -- Close a position ----------------------------------------------------

    def _close(self, open_positions, symbol, exit_price, date_str, reason, cpr_type):
        pos = open_positions.pop(symbol, None)
        if not pos:
            return
        pnl = (exit_price - pos.entry) * pos.qty if pos.side == "long" \
              else (pos.entry - exit_price) * pos.qty
        self.equity += pnl
        self.trades.append({
            "symbol":      symbol,
            "date":        date_str,
            "side":        pos.side,
            "entry":       round(pos.entry, 4),
            "exit":        round(exit_price, 4),
            "qty":         pos.qty,
            "pnl":         round(pnl, 2),
            "score":       pos.score,
            "exit_reason": reason,
            "cpr_type":    cpr_type,
            "win":         pnl > 0,
        })

    # -- Performance report --------------------------------------------------

    def _report(self):
        if not self.trades:
            log.warning("  No trades generated - try widening the date range or "
                        "loosening VPOC_TOUCH_TOL_PCT.")
            return

        df = pd.DataFrame(self.trades)
        df["win"] = df["win"].astype(bool)

        total  = len(df)
        wins   = df[df["win"]]
        losses = df[~df["win"]]
        wr     = len(wins) / total * 100
        gp     = wins["pnl"].sum()
        gl     = abs(losses["pnl"].sum())
        pf     = gp / gl if gl else float("inf")
        avg_w  = wins["pnl"].mean()   if not wins.empty   else 0
        avg_l  = losses["pnl"].mean() if not losses.empty else 0
        avg_rr = abs(avg_w / avg_l)   if avg_l else float("inf")
        net    = df["pnl"].sum()
        ret    = net / CONFIG["INITIAL_CAPITAL"] * 100

        eq     = pd.DataFrame(self.equity_curve).set_index("date")["equity"]
        peak   = eq.cummax()
        dd     = (eq - peak) / peak * 100
        max_dd = dd.min()
        ret_d  = eq.pct_change().dropna()
        sharpe = (ret_d.mean() / ret_d.std() * np.sqrt(252)) if ret_d.std() > 0 else 0

        sep = "-" * 62
        log.info("")
        log.info("=" * 62)
        log.info("  GRAVITATION FRAMEWORK - BACKTEST RESULTS")
        log.info("=" * 62)
        log.info(f"  Period         : {CONFIG['START_DATE']} -> {CONFIG['END_DATE']}")
        log.info(f"  Initial Capital: ${CONFIG['INITIAL_CAPITAL']:>12,.2f}")
        log.info(f"  Final Equity   : ${self.equity:>12,.2f}")
        log.info(f"  Net P&L        : ${net:>+12,.2f}")
        log.info(f"  Total Return   : {ret:>+10.2f}%")
        log.info(sep)
        log.info(f"  Total Trades   : {total}")
        log.info(f"  Win Rate       : {wr:.1f}%")
        log.info(f"  Profit Factor  : {pf:.2f}")
        log.info(f"  Avg Win        : ${avg_w:>+,.2f}")
        log.info(f"  Avg Loss       : ${avg_l:>+,.2f}")
        log.info(f"  Avg R:R        : {avg_rr:.2f}:1")
        log.info(sep)
        log.info(f"  Max Drawdown   : {max_dd:.2f}%")
        log.info(f"  Sharpe Ratio   : {sharpe:.2f}")
        log.info(sep)
        log.info("  PER-TICKER BREAKDOWN:")
        log.info(f"  {'Symbol':<8} {'Trades':>6}  {'Net P&L':>12}  {'Win %':>7}")
        for sym, g in df.groupby("symbol"):
            log.info(f"  {sym:<8} {len(g):>6}  ${g['pnl'].sum():>+11,.2f}  "
                     f"{g['win'].mean()*100:>6.1f}%")
        log.info(sep)
        log.info("  TRADES BY CONFLUENCE SCORE:")
        log.info(f"  {'Score':>6}  {'Trades':>6}  {'Win %':>7}  {'Net P&L':>12}")
        for sc in sorted(df["score"].unique(), reverse=True):
            sg = df[df["score"] == sc]
            log.info(f"  {sc:>6}  {len(sg):>6}  {sg['win'].mean()*100:>6.1f}%  "
                     f"${sg['pnl'].sum():>+11,.2f}")
        log.info("=" * 62)

        out_csv = "gravitation_backtest_results.csv"
        df.to_csv(out_csv, index=False)
        log.info(f"  Trade log saved -> {out_csv}")

        self._plot(eq, dd, df)

    def _plot(self, eq, dd, df):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                 gridspec_kw={"height_ratios": [3, 1.5, 1]})
        fig.patch.set_facecolor("#0d1117")
        for ax in axes:
            ax.set_facecolor("#0d1117")
            ax.tick_params(colors="#c9d1d9")
            ax.spines[:].set_color("#30363d")
            ax.yaxis.label.set_color("#c9d1d9")
            ax.title.set_color("#e6edf3")

        dates = pd.to_datetime(eq.index)

        ax1 = axes[0]
        ax1.plot(dates, eq.values, color="#58a6ff", linewidth=1.8)
        ax1.fill_between(dates, CONFIG["INITIAL_CAPITAL"], eq.values,
                         where=eq.values >= CONFIG["INITIAL_CAPITAL"],
                         alpha=0.15, color="#3fb950")
        ax1.fill_between(dates, CONFIG["INITIAL_CAPITAL"], eq.values,
                         where=eq.values < CONFIG["INITIAL_CAPITAL"],
                         alpha=0.15, color="#f85149")
        ax1.axhline(CONFIG["INITIAL_CAPITAL"], color="#8b949e", linestyle="--", linewidth=0.8)
        ax1.set_title("GRAVITATION FRAMEWORK - EQUITY CURVE (Polygon.io)",
                      fontsize=12, fontweight="bold", pad=10)
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())

        ax2 = axes[1]
        dd_dates = pd.to_datetime(dd.index)
        ax2.fill_between(dd_dates, dd.values, 0, color="#f85149", alpha=0.5)
        ax2.plot(dd_dates, dd.values, color="#f85149", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_title("Drawdown", fontsize=10, pad=6)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        ax3 = axes[2]
        df2 = df.copy()
        df2["month"] = pd.to_datetime(df2["date"]).dt.to_period("M")
        monthly = df2.groupby("month")["pnl"].sum()
        colors  = ["#3fb950" if v > 0 else "#f85149" for v in monthly.values]
        ax3.bar([str(m) for m in monthly.index], monthly.values,
                color=colors, width=0.6, edgecolor="#30363d")
        ax3.axhline(0, color="#8b949e", linewidth=0.7)
        ax3.set_ylabel("Monthly P&L ($)")
        ax3.set_title("Monthly P&L", fontsize=10, pad=6)
        ax3.set_xticks(range(len(monthly)))
        ax3.set_xticklabels([str(m) for m in monthly.index], rotation=45, ha="right", fontsize=7)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        plt.tight_layout(pad=2.0)
        out_png = "gravitation_equity_curve.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        log.info(f"  Equity curve saved -> {out_png}")


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    bt = GravitationBacktester()
    bt.run()