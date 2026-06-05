"""
Feature engineering.

growth_z  — composite of three independent daily/weekly FRED signals:
  1. T10Y2Y  yield curve spread  (40%) — most empirically validated recession
             predictor; inverts before every post-WW2 US recession.
  2. HY OAS  credit spread       (35%) — credit markets price growth risk
             before equities; widening = financial stress building.
  3. ICSA    jobless claims 4w Δ (25%) — most timely labour signal; claims
             rise before unemployment rate does.
  Copper is kept as an auxiliary column for the dashboard but no longer
  drives the regime — it is too China-supply-driven to be a clean proxy.

inflation_z — z-score of 60d momentum of breakeven inflation (T10YIE).
              Breakeven is the market's real-time inflation expectation,
              updated daily with zero publication lag.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window // 2).mean()
    sd = s.rolling(window, min_periods=window // 2).std().replace(0, np.nan)
    return ((s - mu) / sd).fillna(0.0)


def build_features(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    w      = cfg.features.zscore_window      # 252 — rolling z-score window
    mom    = cfg.features.momentum_window    # 60  — inflation momentum lookback
    ry_w   = cfg.features.real_yield_window  # 20  — real yield trend window
    cw     = cfg.features.claims_window      # 4   — weeks to diff jobless claims
    gw     = cfg.features.growth_weights     # composite weights dict

    f = pd.DataFrame(index=raw.index)

    f["gold"]   = raw["gold"]
    f["silver"] = raw["silver"]

    # ------------------------------------------------------------------
    # GROWTH AXIS — composite of yield curve + credit spread + claims
    # ------------------------------------------------------------------

    # 1. Yield curve: T10Y2Y level — positive = normal, negative = inverted
    #    Already in spread units; z-score over rolling window.
    yc_raw = raw["t10y2y"].ffill()
    yc_z   = _zscore(yc_raw, w)

    # 2. HY credit spread: wider spread = stress = bad growth → invert
    #    Use the level (not momentum) — the level tells you current stress.
    hy_raw = raw["hy_oas"].ffill()
    hy_z   = _zscore(-hy_raw, w)   # inverted: high spread → negative z

    # 3. Jobless claims: weekly data, ffill to daily.
    #    4-week % change removes seasonal effects; invert (more claims = worse).
    #    cw weeks × 5 business days = rolling diff period.
    claims_raw  = raw["icsa"].ffill()
    claims_chg  = claims_raw.pct_change(cw * 5).fillna(0)
    claims_z    = _zscore(-claims_chg, w)  # inverted: rising claims → negative z

    # Weighted composite — weights from config (sum = 1.0)
    w_yc  = gw["yield_curve"]
    w_hy  = gw["hy_spread"]
    w_cl  = gw["jobless"]
    f["growth_z"] = w_yc * yc_z + w_hy * hy_z + w_cl * claims_z

    # Component columns — visible on dashboard for transparency
    f["yc_z"]      = yc_z
    f["hy_z"]      = hy_z
    f["claims_z"]  = claims_z

    # ------------------------------------------------------------------
    # INFLATION AXIS — level (60%) + momentum (40%)
    #
    # Pure momentum was answering "is inflation accelerating?" which made
    # inflation_z go negative even when breakevens sat at 2.36% — clearly
    # an inflationary environment. Blending in the level fixes this:
    #   level component    → "are we in an inflationary environment?"
    #   momentum component → "is it getting worse or better?"
    # ------------------------------------------------------------------
    t10yie = raw["t10yie"].ffill()
    infl_level   = _zscore(t10yie, w)                          # level z-score
    infl_mom     = _zscore(t10yie.pct_change(mom).fillna(0), w)  # momentum z-score
    f["inflation_z"]     = 0.60 * infl_level + 0.40 * infl_mom
    f["infl_level_z"]    = infl_level   # stored for dashboard transparency
    f["infl_mom_z"]      = infl_mom

    # ------------------------------------------------------------------
    # REAL YIELD — gold's master variable
    # ------------------------------------------------------------------
    f["real_yield"]     = raw["dfii10"].ffill()
    f["real_yield_chg"] = f["real_yield"].diff(ry_w).fillna(0)

    # ------------------------------------------------------------------
    # AUXILIARY — kept for dashboard / context, not used in regime logic
    # ------------------------------------------------------------------
    f["gold_silver_ratio"] = (
        (raw["gold"] / raw["silver"])
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
    )
    f["copper"] = raw["copper"].ffill()   # legacy context column
    f["vix"]    = raw["vix"].ffill()
    f["oil"]    = raw["oil"].ffill()

    return f.dropna(subset=["growth_z", "inflation_z"])
