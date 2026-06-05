"""
Rule-based regime classifier (the PRIMARY signal).

Maps the (growth_z, inflation_z) plane to the four metal regimes. Fully
transparent: you can read off why any day was classified the way it was.
A small deadband keeps the axes from flickering around zero.
"""
from __future__ import annotations
import pandas as pd

# (growth_up, inflation_up) -> regime
_QUADRANT = {
    (True, True): "Reflation",
    (True, False): "Goldilocks",
    (False, True): "Stagflation",
    (False, False): "Deflation",
}


def classify_rules(feats: pd.DataFrame, cfg) -> pd.Series:
    db = cfg.features.deadband
    g, i = feats["growth_z"], feats["inflation_z"]

    def label(row_g, row_i, prev):
        gu = prev[0] if abs(row_g) < db else row_g > 0
        iu = prev[1] if abs(row_i) < db else row_i > 0
        return gu, iu

    regimes, prev = [], (g.iloc[0] > 0, i.iloc[0] > 0)
    for gg, ii in zip(g, i):
        prev = label(gg, ii, prev)
        regimes.append(_QUADRANT[prev])
    return pd.Series(regimes, index=feats.index, name="candidate")
