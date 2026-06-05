"""
Orchestration. Pure batch job: collect -> features -> rules -> confirm ->
HMM overlay -> bias -> store. In production this becomes a scheduled run
(cron / Airflow / a Cloud Run job) writing to the SQL store; the dashboard
just reads what this produced.

Run:  python -m src.pipeline
"""
from __future__ import annotations
import pandas as pd

from src.config import load_config
from src.data.collectors import get_collector
from src.data.storage import Store
from src.features.engineering import build_features
from src.regime.rules import classify_rules
from src.regime.confirmation import apply_confirmation
from src.regime.hmm_model import fit_hmm_overlay
from src.regime.bias import compute_bias


def run(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    store = Store(cfg)

    raw = get_collector(cfg).collect()
    store.save("raw", raw)

    feats = build_features(raw, cfg)
    store.save("features", feats)

    candidate = classify_rules(feats, cfg)
    regime = apply_confirmation(candidate, cfg)
    bias = compute_bias(regime, feats, cfg)

    overlay = fit_hmm_overlay(feats, cfg)
    result = pd.concat([candidate, bias, feats[["gold", "silver",
                        "growth_z", "yc_z", "hy_z", "claims_z",
                        "inflation_z", "infl_level_z", "infl_mom_z",
                        "real_yield", "real_yield_chg",
                        "gold_silver_ratio"]]], axis=1)
    if overlay is not None:
        result = pd.concat([result, overlay], axis=1)

    store.save("regime", result)

    flips = (regime != regime.shift()).sum()
    print(f"[pipeline] {cfg.data.source} | {len(result)} days "
          f"({result.index.min().date()}..{result.index.max().date()})")
    print(f"[pipeline] committed regime flips: {flips} "
          f"(~1 every {len(result)//max(flips,1)} trading days)")
    print(f"[pipeline] latest: {result.iloc[-1]['regime']} | "
          f"gold {result.iloc[-1]['gold_bias']} | "
          f"silver {result.iloc[-1]['silver_bias']}")
    return result


if __name__ == "__main__":
    run()
