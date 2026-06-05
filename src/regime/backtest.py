"""
Backtest helpers for the dashboard's historical-comparison section.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def regime_conditional_returns(df: pd.DataFrame, horizons=(1, 5, 20)) -> pd.DataFrame:
    """Mean forward returns for gold and silver by committed regime."""
    rows = []
    for h in horizons:
        gold_fwd   = df["gold"].pct_change(h).shift(-h) * 100
        silver_fwd = df["silver"].pct_change(h).shift(-h) * 100
        for reg, grp in df.groupby("regime"):
            idx = grp.index
            g = gold_fwd.reindex(idx).dropna()
            s = silver_fwd.reindex(idx).dropna()
            rows.append({
                "regime":           reg,
                "horizon":          f"{h}d",
                "gold_avg_%":       round(g.mean(), 2),
                "gold_bull_%":      round((g > 0).mean() * 100, 1),
                "gold_bear_%":      round((g < 0).mean() * 100, 1),
                "silver_avg_%":     round(s.mean(), 2),
                "silver_bull_%":    round((s > 0).mean() * 100, 1),
                "silver_bear_%":    round((s < 0).mean() * 100, 1),
                "n_days":           len(g),
            })
    return (
        pd.DataFrame(rows)
        .sort_values(["horizon", "regime"])
        .reset_index(drop=True)
    )


def regime_bias_scores(df: pd.DataFrame, horizon: int = 20) -> dict:
    """
    Derive bias scores and bull/bear probabilities directly from historical
    return distributions. Replaces the hardcoded config values.

    Score mapping (based on bull% at the given horizon):
        bull% >= 65  →  +2  Strong Long
        bull% >= 55  →  +1  Long
        bull% >= 45  →  0   Neutral
        bull% >= 35  →  -1  Short
        bull%  < 35  →  -2  Strong Short
    """
    LABELS = {2: "Strong Long", 1: "Long", 0: "Neutral", -1: "Short", -2: "Strong Short"}

    def score(bull_pct: float) -> int:
        if bull_pct >= 65: return  2
        if bull_pct >= 55: return  1
        if bull_pct >= 45: return  0
        if bull_pct >= 35: return -1
        return -2

    gold_fwd   = df["gold"].pct_change(horizon).shift(-horizon) * 100
    silver_fwd = df["silver"].pct_change(horizon).shift(-horizon) * 100

    result = {}
    for reg, grp in df.groupby("regime"):
        idx = grp.index
        g = gold_fwd.reindex(idx).dropna()
        s = silver_fwd.reindex(idx).dropna()

        g_bull = round((g > 0).mean() * 100, 1)
        s_bull = round((s > 0).mean() * 100, 1)
        g_bear = round((g < 0).mean() * 100, 1)
        s_bear = round((s < 0).mean() * 100, 1)
        g_sc   = score(g_bull)
        s_sc   = score(s_bull)

        result[reg] = {
            "gold_score":    g_sc,
            "gold_bias":     LABELS[g_sc],
            "gold_bull_pct": g_bull,
            "gold_bear_pct": g_bear,
            "gold_avg_ret":  round(g.mean(), 2),
            "gold_std_ret":  round(g.std(), 2),
            "silver_score":    s_sc,
            "silver_bias":     LABELS[s_sc],
            "silver_bull_pct": s_bull,
            "silver_bear_pct": s_bear,
            "silver_avg_ret":  round(s.mean(), 2),
            "silver_std_ret":  round(s.std(), 2),
            "n_days": len(g),
        }
    return result


def regime_transition_status(df: pd.DataFrame, cfg) -> dict:
    """
    Calculate how close the current candidate is to flipping the committed regime.

    Returns a dict with:
      candidate        — what the rules layer is saying right now
      committed        — the current committed regime
      streak           — consecutive days the current candidate has held
      confirm_days     — days needed to confirm a flip
      streak_needed    — remaining streak days before condition is met
      days_since_flip  — trading days since the last committed regime change
      min_dwell        — minimum dwell days required before any flip is allowed
      dwell_needed     — remaining dwell days before flip is allowed
      can_flip         — True if dwell condition is already satisfied
      days_to_flip     — max(streak_needed, dwell_needed) — worst-case days left
      flipping_to      — regime that would be committed if flip happens (=candidate)
    """
    confirm = cfg.regime.confirm_days
    min_dwell = cfg.regime.min_dwell

    # current candidate and committed
    candidate = df["candidate"].iloc[-1]
    committed = df["regime"].iloc[-1]

    # count consecutive days where candidate == current candidate (from the end)
    streak = 0
    for c in reversed(df["candidate"].tolist()):
        if c == candidate:
            streak += 1
        else:
            break

    # days since last committed regime flip
    flips = df[df["regime"] != df["regime"].shift()]
    last_flip_date = flips.index[-1] if len(flips) > 0 else df.index[0]
    days_since_flip = (df.index > last_flip_date).sum()  # rows after last flip

    streak_needed = max(0, confirm   - streak)
    dwell_needed  = max(0, min_dwell - days_since_flip)
    days_to_flip  = max(streak_needed, dwell_needed)
    can_flip      = dwell_needed == 0

    return {
        "candidate":       candidate,
        "committed":       committed,
        "streak":          streak,
        "confirm_days":    confirm,
        "streak_needed":   streak_needed,
        "days_since_flip": days_since_flip,
        "min_dwell":       min_dwell,
        "dwell_needed":    dwell_needed,
        "can_flip":        can_flip,
        "days_to_flip":    days_to_flip,
        "flipping_to":     candidate if candidate != committed else None,
    }


def regime_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Every committed-regime episode, most recent first."""
    reg    = df["regime"]
    blocks = (reg != reg.shift()).cumsum()
    rows   = []

    for _, grp in df.groupby(blocks):
        start = grp.index[0]
        end   = grp.index[-1]
        rows.append({
            "regime":          grp["regime"].iloc[0],
            "start":           start.date(),
            "end":             end.date(),
            "days":            len(grp),
            "gold_ret_%":      round((grp["gold"].iloc[-1]   / grp["gold"].iloc[0]   - 1) * 100, 1),
            "silver_ret_%":    round((grp["silver"].iloc[-1] / grp["silver"].iloc[0] - 1) * 100, 1),
            "avg_growth_z":    round(grp["growth_z"].mean(), 2),
            "avg_inflation_z": round(grp["inflation_z"].mean(), 2),
            "avg_real_yield":  round(grp["real_yield"].mean(), 2),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("start", ascending=False)
        .reset_index(drop=True)
    )
