"""
Directional bias.

Two-stage scoring:
  1. BASE SCORE — derived from historical bull% at 20-day horizon.
     If enough history exists (>= 60 days per regime) the score is computed
     from data and the config values are ignored. This prevents the config
     from drifting out of sync with reality.

     Bull% → score mapping:
       >= 65%  →  +2  Strong Long
       >= 55%  →  +1  Long
       >= 45%  →   0  Neutral
       >= 35%  →  -1  Short
        < 35%  →  -2  Strong Short

  2. REAL-YIELD OVERLAY — gold gets ±1 adjustment based on the 20-day
     direction of real yields (TIPS). Falling real yields = gold tailwind.
     This is an independent daily signal layered on top of the base score.
"""
from __future__ import annotations
import pandas as pd

_LABELS = {2: "Strong Long", 1: "Long", 0: "Neutral", -1: "Short", -2: "Strong Short"}
_MIN_HISTORY = 60   # minimum days per regime before trusting data-driven score


def _clip(x) -> int:
    return max(-2, min(2, int(round(x))))


def _bull_to_score(bull_pct: float) -> int:
    if   bull_pct >= 65: return  2
    elif bull_pct >= 55: return  1
    elif bull_pct >= 45: return  0
    elif bull_pct >= 35: return -1
    else:                return -2


def _derive_base_scores(
    regime: pd.Series,
    feats: pd.DataFrame,
    horizon: int = 20,
) -> dict[str, dict[str, int]]:
    """
    Compute data-driven base scores from the gold/silver price history
    available at pipeline time. Uses a 252-day expanding window so early
    history uses config fallback until enough data accumulates.
    """
    gold   = feats["gold"]
    silver = feats["silver"]
    scores = {}
    for reg in regime.unique():
        idx    = regime[regime == reg].index
        g_fwd  = gold.pct_change(horizon).shift(-horizon).reindex(idx).dropna()
        s_fwd  = silver.pct_change(horizon).shift(-horizon).reindex(idx).dropna()
        if len(g_fwd) < _MIN_HISTORY:
            scores[reg] = None          # not enough data — use config fallback
        else:
            scores[reg] = {
                "gold":   _bull_to_score((g_fwd > 0).mean() * 100),
                "silver": _bull_to_score((s_fwd > 0).mean() * 100),
            }
    return scores


def compute_bias(
    regime: pd.Series,
    feats: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    overlay    = cfg.bias.real_yield_overlay
    ry_chg     = feats["real_yield_chg"]
    config_base = cfg.bias.base          # fallback when history is thin

    # derive data-driven base scores once for the full history
    data_scores = _derive_base_scores(regime, feats)

    rows = []
    for dt, reg in regime.items():
        # prefer data-driven; fall back to config if not enough history
        reg_scores = data_scores.get(reg) or {}
        g = reg_scores.get("gold",   config_base[reg]["gold"])
        s = reg_scores.get("silver", config_base[reg]["silver"])

        if overlay:
            g += -1 if ry_chg.get(dt, 0) > 0 else 1

        g, s = _clip(g), _clip(s)
        rows.append({
            "regime":       reg,
            "gold_score":   g,
            "silver_score": s,
            "gold_bias":    _LABELS[g],
            "silver_bias":  _LABELS[s],
        })
    return pd.DataFrame(rows, index=regime.index)
