"""
HMM overlay (optional risk-state signal).

Fits a Gaussian HMM on realised vol + real-yield change to label days as
risk-off / neutral / risk-on. Result is appended as `hmm_state` and
`hmm_confidence` columns. If hmmlearn is not installed or HMM is disabled
in config, returns None and the pipeline continues without it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fit_hmm_overlay(feats: pd.DataFrame, cfg) -> pd.DataFrame | None:
    if not cfg.regime.hmm.enabled:
        return None
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("[hmm] hmmlearn not installed — skipping overlay")
        return None

    n = cfg.regime.hmm.n_states
    seed = cfg.regime.hmm.seed

    # features: real-yield change + gold daily return + vix (if present)
    gold_ret = feats["gold"].pct_change().fillna(0)
    ry_chg = feats["real_yield_chg"].fillna(0)
    cols = np.column_stack([gold_ret.values, ry_chg.values])
    if "vix" in feats.columns:
        cols = np.column_stack([cols, feats["vix"].fillna(feats["vix"].mean()).values])

    model = GaussianHMM(n_components=n, covariance_type="diag",
                        n_iter=200, random_state=seed)
    model.fit(cols)
    states = model.predict(cols)
    log_prob = model.predict_proba(cols)
    confidence = log_prob.max(axis=1)

    # label states by mean gold return: highest = risk-on
    mean_ret = [gold_ret.values[states == s].mean() for s in range(n)]
    rank = np.argsort(mean_ret)  # 0=risk-off, n-1=risk-on
    label_map = {rank[i]: i for i in range(n)}
    named = pd.Categorical(
        [["risk_off", "neutral", "risk_on"][label_map[s]] for s in states]
        if n == 3 else [str(label_map[s]) for s in states]
    )

    return pd.DataFrame(
        {"hmm_state": named, "hmm_confidence": confidence},
        index=feats.index,
    )
