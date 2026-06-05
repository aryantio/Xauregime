"""
Daily dashboard.  Run:  streamlit run dashboard/app.py

Reads only from the store (what the pipeline produced). Four sections:
  1. Current regime + data-driven bias with bull/bear probabilities
  2. Driver gauges (growth composite, inflation, real-yield, GSR)
  3. Regime timeline overlaid on gold & silver price
  4. Historical comparison: probability table + episode table
"""
from __future__ import annotations
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.data.storage import Store
from src.regime.backtest import (
    regime_conditional_returns,
    regime_bias_scores,
    regime_transition_status,
    regime_episodes,
)

REGIME_COLORS = {
    "Stagflation": "#1D9E75", "Reflation": "#639922",
    "Goldilocks":  "#D85A30", "Deflation": "#888780",
}

st.set_page_config(page_title="Gold/Silver Regime Detector", layout="wide")
cfg = load_config()


@st.cache_data(ttl=3600)
def load():
    return Store(cfg).load("regime")


try:
    df = load()
except FileNotFoundError:
    st.error("No data yet. Run `python -m src.pipeline` first.")
    st.stop()

latest  = df.iloc[-1]
reg_now = latest["regime"]

# derive bias from history (20d horizon)
bias_table = regime_bias_scores(df, horizon=20)
bias_now   = bias_table.get(reg_now, {})

# ---------- 1. headline ----------
st.title("Gold / Silver macro regime")
st.caption(f"As of {df.index[-1].date()}  ·  source: {cfg.data.source}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Committed regime", reg_now)
# prefer FOMC-adjusted bias if available, fall back to base
gold_bias_show   = latest.get("gold_bias_adj",   latest["gold_bias"])
silver_bias_show = latest.get("silver_bias_adj", latest["silver_bias"])
gold_delta   = f"base: {latest['gold_bias']}"   if "gold_bias_adj"   in latest.index else ""
silver_delta = f"base: {latest['silver_bias']}" if "silver_bias_adj" in latest.index else ""
c2.metric("Gold bias",   gold_bias_show,   delta=gold_delta   or None)
c3.metric("Silver bias", silver_bias_show, delta=silver_delta or None)
if "hmm_confidence" in df.columns:
    c4.metric("HMM confidence", f"{latest['hmm_confidence']*100:.0f}%")

# ---------- transition direction ----------
ts = regime_transition_status(df, cfg)

if ts["flipping_to"] and ts["flipping_to"] != ts["committed"]:
    # actively trying to flip — show progress bar
    st.subheader("⏳ Regime transition in progress")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Candidate (direction)",  ts["candidate"])
    t2.metric("Streak so far",
              f"{ts['streak']} / {ts['confirm_days']} days",
              f"{ts['streak_needed']} more needed" if ts['streak_needed'] > 0 else "✅ streak met")
    t3.metric("Dwell since last flip",
              f"{ts['days_since_flip']} / {ts['min_dwell']} days",
              f"{ts['dwell_needed']} more needed" if ts['dwell_needed'] > 0 else "✅ dwell met")
    t4.metric("Earliest possible flip",
              f"~{ts['days_to_flip']} trading days",
              "if candidate holds every day")

    # progress bars
    streak_pct = min(100, int(ts["streak"] / ts["confirm_days"] * 100))
    dwell_pct  = min(100, int(ts["days_since_flip"] / ts["min_dwell"] * 100))
    st.markdown(f"**Streak progress → {ts['candidate']}**")
    st.progress(streak_pct, text=f"{ts['streak']} of {ts['confirm_days']} days confirmed")
    st.markdown("**Dwell lock progress**")
    st.progress(dwell_pct,  text=f"{ts['days_since_flip']} of {ts['min_dwell']} days elapsed since last flip")

    if ts["days_to_flip"] == 0:
        st.success(f"🚨 Both conditions met — committed regime will flip to **{ts['candidate']}** on next pipeline run!")
    elif not ts["can_flip"]:
        st.info(f"Dwell lock active for {ts['dwell_needed']} more day(s) — even a full streak cannot flip the regime yet.")
else:
    st.subheader("📍 Regime direction")
    st.success(f"Candidate **{ts['candidate']}** agrees with committed regime — no flip in progress. "
               f"Streak: {ts['streak']} days.")

st.divider()

# ── FOMC overlay panel ────────────────────────────────────────────────────
if "fomc_score" in df.columns:
    fomc_score = latest.get("fomc_score", 0)
    fomc_label = latest.get("fomc_label", "Neutral")
    FOMC_COLOR = {
        "Very Dovish": "🟢", "Dovish": "🟡",
        "Neutral": "⚪",
        "Hawkish": "🟠", "Very Hawkish": "🔴",
    }
    icon = FOMC_COLOR.get(fomc_label, "⚪")
    st.subheader(f"{icon} FOMC stance: {fomc_label}  ({fomc_score:+d})")

    fa, fb, fc, fd = st.columns(4)
    fa.metric("FOMC score",        f"{fomc_score:+d} / ±2")
    fb.metric("Gold bias (base)",  latest["gold_bias"])
    fc.metric("Gold bias (FOMC-adjusted)", latest.get("gold_bias_adj", latest["gold_bias"]))

    # show last 5 FOMC meetings from DB
    try:
        fomc_hist = Store(cfg).load("fomc_scores").sort_index(ascending=False).head(6)
        fomc_hist.index = fomc_hist.index.date
        with st.expander("Last 6 FOMC meetings"):
            st.dataframe(
                fomc_hist[["score", "label", "reasoning"]],
                use_container_width=True,
            )
    except Exception:
        pass
    st.divider()

# --- bull/bear probability gauges for current regime ---
st.subheader(f"Probability outlook — {reg_now} regime (20-day horizon)")
p1, p2, p3, p4 = st.columns(4)

g_bull = bias_now.get("gold_bull_pct", 0)
g_bear = bias_now.get("gold_bear_pct", 0)
s_bull = bias_now.get("silver_bull_pct", 0)
s_bear = bias_now.get("silver_bear_pct", 0)

p1.metric("🟡 Gold bull probability",  f"{g_bull:.1f}%",
          f"avg +{bias_now.get('gold_avg_ret',0):.2f}%  std ±{bias_now.get('gold_std_ret',0):.2f}%")
p2.metric("🔴 Gold bear probability",  f"{g_bear:.1f}%")
p3.metric("⚪ Silver bull probability", f"{s_bull:.1f}%",
          f"avg +{bias_now.get('silver_avg_ret',0):.2f}%  std ±{bias_now.get('silver_std_ret',0):.2f}%")
p4.metric("🔴 Silver bear probability", f"{s_bear:.1f}%")

# visual probability bars
fig_prob = go.Figure()
for metal, bull, bear, color in [
    ("Gold",   g_bull, g_bear, "#BA7517"),
    ("Silver", s_bull, s_bear, "#888780"),
]:
    fig_prob.add_trace(go.Bar(
        name=f"{metal} Bull", x=[metal], y=[bull],
        marker_color=color, text=f"{bull:.1f}%", textposition="inside",
    ))
    fig_prob.add_trace(go.Bar(
        name=f"{metal} Bear", x=[metal], y=[bear],
        marker_color="#c0392b", text=f"{bear:.1f}%", textposition="inside",
    ))
fig_prob.update_layout(
    barmode="group", height=220, showlegend=True,
    margin=dict(t=10, b=10),
    yaxis=dict(title="Probability %", range=[0, 100]),
)
st.plotly_chart(fig_prob, use_container_width=True)

# ---------- 2. drivers ----------
st.subheader("Drivers")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Growth axis (z)",    f"{latest['growth_z']:+.2f}")
d2.metric("Inflation axis (z)", f"{latest['inflation_z']:+.2f}")
ry = latest["real_yield_chg"]
d3.metric("Real-yield 20d chg", f"{ry:+.2f}",
          "falling = gold tailwind" if ry < 0 else "rising = headwind")
d4.metric("Gold/Silver ratio",  f"{latest['gold_silver_ratio']:.1f}")

with st.expander("Growth composite breakdown"):
    g1, g2, g3 = st.columns(3)
    g1.metric("Yield curve z (40%)",      f"{latest['yc_z']:+.2f}",
              "normal" if latest["yc_z"] > 0 else "flattening/inverted")
    g2.metric("HY credit spread z (35%)", f"{latest['hy_z']:+.2f}",
              "tight = growth OK" if latest["hy_z"] > 0 else "widening = stress")
    g3.metric("Jobless claims z (25%)",   f"{latest['claims_z']:+.2f}",
              "falling = labour strong" if latest["claims_z"] > 0 else "rising = labour weakening")

with st.expander("Inflation composite breakdown"):
    i1, i2, i3 = st.columns(3)
    i1.metric("Inflation z (composite)",  f"{latest['inflation_z']:+.2f}",
              "above avg" if latest["inflation_z"] > 0 else "below avg")
    i2.metric("Level z (60% weight)",     f"{latest['infl_level_z']:+.2f}",
              "breakeven above historical avg" if latest["infl_level_z"] > 0
              else "breakeven below historical avg")
    i3.metric("Momentum z (40% weight)",  f"{latest['infl_mom_z']:+.2f}",
              "inflation accelerating" if latest["infl_mom_z"] > 0
              else "inflation decelerating")

# ---------- 3. timeline ----------
st.subheader("Regime history vs price")
reg    = df["regime"]
blocks = (reg != reg.shift()).cumsum()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["gold"],
                         name="Gold", line=dict(color="#BA7517")))
fig.add_trace(go.Scatter(x=df.index, y=df["silver"],
                         name="Silver", yaxis="y2", line=dict(color="#888780")))
for _, b in df.groupby(blocks):
    fig.add_vrect(
        x0=b.index.min(), x1=b.index.max(),
        fillcolor=REGIME_COLORS.get(b["regime"].iloc[0], "#ccc"),
        opacity=0.12, line_width=0,
    )
fig.update_layout(
    height=420,
    yaxis=dict(title="Gold"),
    yaxis2=dict(title="Silver", overlaying="y", side="right"),
    legend=dict(orientation="h"), margin=dict(t=10),
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Shaded bands = committed regime  ·  "
           + "  ".join(k for k in REGIME_COLORS))

# ---------- 4. historical comparison ----------
st.subheader("Historical comparison")

# --- probability table across all regimes ---
st.markdown("**Bull / bear probability by regime (20-day horizon) — data-driven**")
prob_rows = []
for reg_name, b in bias_table.items():
    prob_rows.append({
        "regime":          reg_name,
        "gold_bias":       b["gold_bias"],
        "gold_bull_%":     b["gold_bull_pct"],
        "gold_bear_%":     b["gold_bear_pct"],
        "gold_avg_ret_%":  b["gold_avg_ret"],
        "gold_std_%":      b["gold_std_ret"],
        "silver_bias":     b["silver_bias"],
        "silver_bull_%":   b["silver_bull_pct"],
        "silver_bear_%":   b["silver_bear_pct"],
        "silver_avg_ret_%":b["silver_avg_ret"],
        "n_days":          b["n_days"],
    })

import pandas as pd
prob_df = (
    pd.DataFrame(prob_rows)
    .sort_values("gold_bull_%", ascending=False)
    .reset_index(drop=True)
)
st.dataframe(prob_df, use_container_width=True, hide_index=True)

# --- forward returns + episodes side by side ---
st.markdown("---")
left, right = st.columns(2)
with left:
    st.markdown("**Forward returns by regime & horizon**")
    st.dataframe(regime_conditional_returns(df), use_container_width=True, hide_index=True)
with right:
    st.markdown("**Regime episodes (most recent first)**")
    st.dataframe(regime_episodes(df), use_container_width=True, hide_index=True)
