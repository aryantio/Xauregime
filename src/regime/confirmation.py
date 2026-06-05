"""
Confirmation / hysteresis.

A new candidate regime must (a) persist for `confirm_days` consecutive
trading days AND (b) the current regime must have been held for at least
`min_dwell` days before any switch is allowed.
"""
from __future__ import annotations
import pandas as pd


def apply_confirmation(candidates: pd.Series, cfg) -> pd.Series:
    confirm = cfg.regime.confirm_days
    dwell = cfg.regime.min_dwell

    committed = []
    current = candidates.iloc[0]
    days_since_commit = dwell
    streak_regime = current
    streak_len = 0

    for c in candidates:
        streak_len = streak_len + 1 if c == streak_regime else 1
        streak_regime = c

        ready = (
            streak_regime != current
            and streak_len >= confirm
            and days_since_commit >= dwell
        )
        if ready:
            current = streak_regime
            days_since_commit = 0
        else:
            days_since_commit += 1
        committed.append(current)

    return pd.Series(committed, index=candidates.index, name="regime")
