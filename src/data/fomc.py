"""
FOMC statement scraper + hawkish/dovish scorer.

Fetches every FOMC statement since 2016 from federalreserve.gov,
scores it with Claude (haiku — cheap, fast) on a -2..+2 scale:
  +2  Very Dovish   → strong gold tailwind
  +1  Dovish        → mild gold tailwind
   0  Neutral
  -1  Hawkish       → mild gold headwind
  -2  Very Hawkish  → strong gold headwind

Scores are stored in the `fomc_scores` table and forward-filled
daily so the bias layer can use them like any other signal.

Usage (called by pipeline):
    from src.data.fomc import build_fomc_overlay
    fomc_daily = build_fomc_overlay(store, cfg, start_date="2016-01-01")
"""
from __future__ import annotations
import os
import time
import re
import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── known FOMC meeting dates (statement release dates) 2016-today ───────────
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# Format: YYYYMMDD — the day the statement was published
FOMC_DATES = [
    # 2016
    "20160127","20160316","20160427","20160615","20160727","20160921","20161102","20161214",
    # 2017
    "20170201","20170315","20170503","20170614","20170726","20170920","20171101","20171213",
    # 2018
    "20180131","20180321","20180502","20180613","20180801","20180926","20181108","20181219",
    # 2019
    "20190130","20190320","20190501","20190619","20190731","20190918","20191030","20191211",
    # 2020
    "20200129","20200303","20200315","20200429","20200610","20200729","20200916","20201105","20201216",
    # 2021
    "20210127","20210317","20210428","20210616","20210728","20210922","20211103","20211215",
    # 2022
    "20220126","20220316","20220504","20220615","20220727","20220921","20221102","20221214",
    # 2023
    "20230201","20230322","20230503","20230614","20230726","20230920","20231101","20231213",
    # 2024
    "20240131","20240320","20240501","20240612","20240731","20240918","20241107","20241218",
    # 2025
    "20250129","20250319","20250507","20250618","20250730","20250917","20251029","20251210",
    # 2026
    "20260128","20260318","20260506",
]

STATEMENT_URL = (
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary{date}a.htm"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FOMCBot/1.0)"}


# ── scraper ──────────────────────────────────────────────────────────────────

def _fetch_statement(date_str: str) -> str | None:
    """Fetch the plain text of one FOMC statement. Returns None on failure."""
    url = STATEMENT_URL.format(date=date_str)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        content = (
            soup.find("div", {"id": "article"})
            or soup.find("div", class_="col-xs-12")
            or soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        )
        if not content:
            return None
        # strip boilerplate vote paragraph + tables
        for tag in content.find_all(["table", "script", "style"]):
            tag.decompose()
        text = content.get_text(" ", strip=True)
        # trim after "Voting for" (vote section adds noise)
        text = re.split(r"Voting for|Voting against", text)[0]
        return text.strip()
    except Exception:
        return None


# ── scorer ───────────────────────────────────────────────────────────────────

def _score_statement(text: str, api_key: str) -> dict:
    """
    Score the statement using Claude haiku.
    Returns {"score": int, "label": str, "reasoning": str}
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are an expert Fed-watcher. Score the following FOMC statement on a hawkish/dovish scale for its implications for GOLD prices.

Scale:
+2 = Very Dovish  (rate cuts likely, accommodation, QE, growth/employment focus)
+1 = Dovish       (leaning toward cuts or holding, concerns about growth)
 0 = Neutral      (balanced, data-dependent, no clear lean)
-1 = Hawkish      (leaning toward hikes or holding high, inflation concern)
-2 = Very Hawkish (aggressive tightening, strong inflation fighting language)

Remember: Hawkish Fed = headwind for gold. Dovish Fed = tailwind for gold.

FOMC Statement:
{text[:3000]}

Respond in this exact format:
SCORE: <integer from -2 to +2>
LABEL: <one of: Very Dovish, Dovish, Neutral, Hawkish, Very Hawkish>
REASONING: <2-3 sentences explaining the key language that drove your score>"""

    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()

    score_match = re.search(r"SCORE:\s*([+-]?\d)", raw)
    label_match = re.search(r"LABEL:\s*(.+)", raw)
    reason_match = re.search(r"REASONING:\s*(.+)", raw, re.DOTALL)

    score = int(score_match.group(1)) if score_match else 0
    score = max(-2, min(2, score))
    label = label_match.group(1).strip() if label_match else "Neutral"
    reasoning = reason_match.group(1).strip() if reason_match else ""

    return {"score": score, "label": label, "reasoning": reasoning}


# ── main entry point ─────────────────────────────────────────────────────────

def build_fomc_overlay(store, cfg, start_date: str = "2016-01-01") -> pd.DataFrame:
    """
    1. Load existing scores from DB (avoid re-scoring what we have).
    2. Scrape + score any missing meetings.
    3. Save updated table to DB.
    4. Return a daily forward-filled Series of FOMC scores aligned to a
       business-day index (for merging into the regime feature set).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")

    # load existing scored meetings from DB (empty DataFrame if table missing)
    try:
        existing = store.load("fomc_scores")
        existing.index = pd.to_datetime(existing.index)
        scored_dates = set(existing.index.strftime("%Y%m%d").tolist())
    except Exception:
        existing = pd.DataFrame(
            columns=["score", "label", "reasoning", "statement_text"]
        )
        scored_dates = set()

    cutoff = pd.Timestamp(start_date)
    pending = [
        d for d in FOMC_DATES
        if d not in scored_dates
        and pd.Timestamp(d) >= cutoff
        and pd.Timestamp(d) <= pd.Timestamp.today()
    ]

    print(f"[fomc] {len(scored_dates)} meetings already scored, "
          f"{len(pending)} pending")

    new_rows = []
    for date_str in pending:
        print(f"[fomc] scoring {date_str} ...", end=" ", flush=True)
        text = _fetch_statement(date_str)
        if not text:
            print("SKIP (could not fetch)")
            continue
        result = _score_statement(text, api_key)
        new_rows.append({
            "date":           pd.Timestamp(date_str),
            "score":          result["score"],
            "label":          result["label"],
            "reasoning":      result["reasoning"],
            "statement_text": text[:2000],
        })
        print(f"{result['label']} ({result['score']:+d})")
        time.sleep(0.3)   # be polite to Fed servers

    if new_rows:
        new_df = pd.DataFrame(new_rows).set_index("date")
        combined = pd.concat([existing, new_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        store.save("fomc_scores", combined)
        print(f"[fomc] saved {len(combined)} total FOMC scores to DB")
    else:
        combined = existing
        print("[fomc] no new meetings to score")

    # ── compute surprise = score change vs prior meeting ────────────────────
    # Surprise is what gold actually reacts to: a pivot from Hawkish→Dovish
    # is a +2 surprise regardless of the absolute level.
    # Decay: the surprise signal fades linearly to 0 over 45 trading days
    # (the typical inter-meeting window) so it doesn't permanently colour
    # the bias the way the raw forward-fill did.
    combined = combined.sort_index()
    combined["surprise"] = combined["score"].diff().fillna(0)

    bday_idx = pd.date_range(start_date, pd.Timestamp.today(), freq="B")

    # forward-fill raw score + label (context / display only)
    daily_score = combined["score"].reindex(bday_idx).ffill().fillna(0).rename("fomc_score")
    daily_label = combined["label"].reindex(bday_idx).ffill().fillna("Neutral").rename("fomc_label")

    # build decayed surprise signal
    # On meeting day: surprise value. Linearly fades to 0 over DECAY_DAYS.
    DECAY_DAYS = 45
    surprise_raw = combined["surprise"].reindex(bday_idx).fillna(0)
    decayed = []
    current_surprise = 0.0
    days_held = 0
    for val in surprise_raw:
        if val != 0:                        # new meeting — reset
            current_surprise = float(val)
            days_held = 0
        fade = max(0.0, 1.0 - days_held / DECAY_DAYS)
        decayed.append(round(current_surprise * fade, 3))
        days_held += 1

    daily_surprise = pd.Series(decayed, index=bday_idx, name="fomc_surprise")

    # surprise label for display
    def _surprise_label(v):
        if   v >  0.5: return "Dovish Surprise"
        elif v >  0:   return "Mild Dovish Surprise"
        elif v == 0:   return "No Surprise"
        elif v > -0.5: return "Mild Hawkish Surprise"
        else:          return "Hawkish Surprise"

    daily_surprise_label = daily_surprise.map(_surprise_label).rename("fomc_surprise_label")

    print(f"[fomc] surprise distribution: "
          + ", ".join(f"{v:.1f}({(daily_surprise==v).sum()}d)"
                      for v in sorted(combined['surprise'].unique())))

    return pd.DataFrame({
        "fomc_score":         daily_score,
        "fomc_label":         daily_label,
        "fomc_surprise":      daily_surprise,
        "fomc_surprise_label": daily_surprise_label,
    })
