"""
Orchestration. Pure batch job:
  collect -> features -> rules -> confirm -> HMM -> bias -> FOMC overlay -> store

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
    regime    = apply_confirmation(candidate, cfg)
    bias      = compute_bias(regime, feats, cfg)
    overlay   = fit_hmm_overlay(feats, cfg)

    result = pd.concat([candidate, bias, feats[["gold", "silver",
                        "growth_z", "yc_z", "hy_z", "claims_z",
                        "inflation_z", "infl_level_z", "infl_mom_z",
                        "real_yield", "real_yield_chg",
                        "gold_silver_ratio"]]], axis=1)

    if overlay is not None:
        result = pd.concat([result, overlay], axis=1)

    # ── FOMC overlay (optional experiment) ──────────────────────────────
    if cfg.fomc.enabled:
        try:
            from src.data.fomc import build_fomc_overlay
            fomc_daily = build_fomc_overlay(
                store, cfg, start_date=cfg.data.start_date
            )
            # align to regime index
            fomc_aligned = fomc_daily.reindex(result.index, method="ffill")
            result = pd.concat([result, fomc_aligned], axis=1)

            # adjust gold/silver score using FOMC signal
            gw = cfg.fomc.gold_weight
            sw = cfg.fomc.silver_weight
            _LABELS = {
                2: "Strong Long", 1: "Long", 0: "Neutral",
                -1: "Short", -2: "Strong Short"
            }
            def _clip(x): return max(-2, min(2, int(round(x))))
            def _adj(base_score_col, weight):
                return result.apply(
                    lambda r: _clip(r[base_score_col] + r["fomc_score"] * weight),
                    axis=1
                )
            result["gold_score_adj"]   = _adj("gold_score",   gw)
            result["silver_score_adj"] = _adj("silver_score", sw)
            result["gold_bias_adj"]    = result["gold_score_adj"].map(_LABELS)
            result["silver_bias_adj"]  = result["silver_score_adj"].map(_LABELS)
            print(f"[fomc] overlay applied — today fomc_score="
                  f"{result.iloc[-1].get('fomc_score', 'n/a')} "
                  f"({result.iloc[-1].get('fomc_label', 'n/a')})")
        except Exception as e:
            print(f"[fomc] SKIPPED — {e}")

    store.save("regime", result)

    flips = (regime != regime.shift()).sum()
    print(f"[pipeline] {cfg.data.source} | {len(result)} days "
          f"({result.index.min().date()}..{result.index.max().date()})")
    print(f"[pipeline] committed regime flips: {flips} "
          f"(~1 every {len(result)//max(flips,1)} trading days)")
    print(f"[pipeline] latest: {result.iloc[-1]['regime']} | "
          f"gold {result.iloc[-1].get('gold_bias_adj', result.iloc[-1]['gold_bias'])} | "
          f"silver {result.iloc[-1].get('silver_bias_adj', result.iloc[-1]['silver_bias'])}")
    return result


if __name__ == "__main__":
    run()
