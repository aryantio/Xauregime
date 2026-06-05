"""
Directional bias. Maps the committed regime to a -2..+2 score for gold and
silver, then applies the real-yield overlay to gold.
"""
from __future__ import annotations
import pandas as pd

_LABELS = {2: "Strong Long", 1: "Long", 0: "Neutral", -1: "Short", -2: "Strong Short"}


def _clip(x: int) -> int:
    return max(-2, min(2, x))


def compute_bias(regime: pd.Series, feats: pd.DataFrame, cfg) -> pd.DataFrame:
    base = cfg.bias.base
    overlay = cfg.bias.real_yield_overlay
    ry_chg = feats["real_yield_chg"]

    rows = []
    for dt, reg in regime.items():
        g = base[reg]["gold"]
        s = base[reg]["silver"]
        if overlay:
            g += -1 if ry_chg.get(dt, 0) > 0 else 1
        g, s = _clip(g), _clip(s)
        rows.append(
            {
                "regime": reg,
                "gold_score": g, "silver_score": s,
                "gold_bias": _LABELS[g], "silver_bias": _LABELS[s],
            }
        )
    return pd.DataFrame(rows, index=regime.index)
