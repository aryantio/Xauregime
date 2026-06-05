"""
Feature engineering.

growth_z  — composite of three independent daily/weekly FRED signals:
  1. T10Y2Y  yield curve spread  (40%)
  2. HY OAS  credit spread       (35%)
  3. ICSA    jobless claims 4w Δ (25%)

inflation_z — three-component composite:
  1. T10YIE level z-score     (60%) — "are we in inflationary territory?"
  2. EMA momentum (12/26)     (25%) — real-time acceleration signal,
                                      smoothest of all tested methods
                                      (1 big-flip vs 10 for pct_change)
  3. OLS slope over 60d       (15%) — uses ALL data points in window,
                                      not just endpoints; zero big-flips

  Why EMA+OLS instead of simple pct_change(60d)?
  - pct_change compares only two endpoints → one stale spike 60 days
    ago contaminates the signal for the full window
  - EMA crossover: exponentially weights recent data, r=-0.188 vs -0.128
  - OLS slope: fits a regression line through all 60 days — immune to
    endpoint noise, r=-0.149, smoothest signal (0 big-flips)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return ((s - mu) / sd).fillna(0.0)


def _ema_momentum(s: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    EMA crossover momentum: (fast_ema - slow_ema) / slow_ema.
    Exponentially weights recent data — reacts quickly to trend changes
    without the endpoint-sensitivity problem of pct_change.
    """
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    return ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)).fillna(0)


def _ols_slope(s: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope over `window` days, annualised.
    Uses ALL data points in the window (not just endpoints) so a single
    outlier 60 days ago cannot dominate the signal.
    """
    def _slope(y: np.ndarray) -> float:
        # y length varies during warm-up — build x to match
        n = len(y)
        mask = ~np.isnan(y)
        if mask.sum() < max(2, n // 2):
            return np.nan
        x = np.arange(n, dtype=float)
        x -= x.mean()                           # centre for numerical stability
        xm, ym = x[mask], y[mask]
        xx = (xm * xm).sum()
        if xx == 0:
            return 0.0
        return float(np.dot(xm, ym) / xx) * 252  # annualised

    return s.rolling(window, min_periods=window // 2).apply(_slope, raw=True)


def build_features(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    w      = cfg.features.zscore_window      # 252 — rolling z-score window
    mom    = cfg.features.momentum_window    # 60  — OLS / pct_change window
    ry_w   = cfg.features.real_yield_window  # 20  — real yield trend window
    cw     = cfg.features.claims_window      # 4   — weeks to diff jobless claims
    gw     = cfg.features.growth_weights     # composite weights dict

    f = pd.DataFrame(index=raw.index)

    f["gold"]   = raw["gold"]
    f["silver"] = raw["silver"]

    # ------------------------------------------------------------------
    # GROWTH AXIS — yield curve + HY credit spread + jobless claims
    # ------------------------------------------------------------------
    yc_raw     = raw["t10y2y"].ffill()
    yc_z       = _zscore(yc_raw, w)

    hy_raw     = raw["hy_oas"].ffill()
    hy_z       = _zscore(-hy_raw, w)

    claims_raw = raw["icsa"].ffill()
    claims_chg = claims_raw.pct_change(cw * 5).fillna(0)
    claims_z   = _zscore(-claims_chg, w)

    w_yc, w_hy, w_cl = gw["yield_curve"], gw["hy_spread"], gw["jobless"]
    f["growth_z"] = w_yc * yc_z + w_hy * hy_z + w_cl * claims_z
    f["yc_z"]     = yc_z
    f["hy_z"]     = hy_z
    f["claims_z"] = claims_z

    # ------------------------------------------------------------------
    # INFLATION AXIS — level (60%) + EMA momentum (25%) + OLS slope (15%)
    #
    # Old method: pct_change(60) compared only two endpoints →
    #   a spike 60 days ago could corrupt the signal for 60 days.
    # New method:
    #   EMA crossover (12/26) — smoothest signal, r=-0.188 vs -0.128
    #   OLS slope (60d)       — uses all points, immune to endpoint noise
    # ------------------------------------------------------------------
    t10yie = raw["t10yie"].ffill()

    infl_level   = _zscore(t10yie, w)
    infl_ema     = _zscore(_ema_momentum(t10yie, fast=12, slow=26), w)
    infl_ols     = _zscore(_ols_slope(t10yie, mom).fillna(0), w)

    f["inflation_z"]   = 0.60 * infl_level + 0.25 * infl_ema + 0.15 * infl_ols
    f["infl_level_z"]  = infl_level
    f["infl_mom_z"]    = infl_ema      # renamed: was pct_change, now EMA crossover
    f["infl_ols_z"]    = infl_ols

    # ------------------------------------------------------------------
    # REAL YIELD — gold's master variable
    # ------------------------------------------------------------------
    f["real_yield"]     = raw["dfii10"].ffill()
    f["real_yield_chg"] = f["real_yield"].diff(ry_w).fillna(0)

    # ------------------------------------------------------------------
    # AUXILIARY
    # ------------------------------------------------------------------
    f["gold_silver_ratio"] = (
        (raw["gold"] / raw["silver"])
        .replace([np.inf, -np.inf], np.nan).ffill()
    )
    f["copper"] = raw["copper"].ffill()
    f["vix"]    = raw["vix"].ffill()
    f["oil"]    = raw["oil"].ffill()

    return f.dropna(subset=["growth_z", "inflation_z"])
