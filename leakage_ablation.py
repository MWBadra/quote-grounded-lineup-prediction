"""How much of the LLM signal is READ from the retrieved text, and how much is
RECALLED from having been trained after the 2025-26 season?

This is the part of the paper most likely to get attacked. The model was trained
after these matches were played, so "it already knew the lineups" is the first
thing anyone will say, and claiming a knowledge cutoff is no answer because
cutoffs are approximate and nobody can check one. Measuring is an answer.

Three arms, same rosters and fixtures, differing only in what the model gets:

  A  sourced    real pre-match articles, quotes verified  (what we propose)
  B  memory     no sources at all, answer from what you know
  C  shuffled   some other club's articles from the same gameweek

How to read it. If B predicts well the model remembers the season, the feature
is contaminated and the headline claim isn't safe. If B lands near chance the
signal really is coming out of the retrieved text. C near chance means it's
reading the sources rather than pattern-matching the fixture, which also kills
the "it's just responding to prompt shape" objection.

Validation window only. That's where every reported metric lives anyway and it
halves the number of calls.

python leakage_ablation.py --live
"""

from __future__ import annotations

import argparse
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from extraction import (ROTATION_LABELS, Extraction, PlayerSignal, TeamSignal,
                        _call_gemini, cache_path, parse, verify)
from llm_keys import ExtractionFailure, with_retry
from retrieval import Corpus, load_cached
from run_pipeline_v2 import rule

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"
OUT = HERE / "reports"
CACHE = {"memory": HERE / ".ablate_memory", "shuffled": HERE / ".ablate_shuffled"}

# The memory arm cannot quote anything, so it is asked for a judgement directly.
# Withholding the quote requirement is deliberate: the question is what the model
# believes unaided, and a quote gate would simply zero every answer and prove
# nothing.
MEMORY_PROMPT = """You are predicting a football team's selection for one fixture.

You will be given the TARGET CLUB, the fixture and date, and the ROSTER.
You are given NO sources. Answer from your own knowledge of this club and season.

Return ONLY raw JSON, no markdown:

{
  "players": [
    {"name": "<exact name from ROSTER>",
     "rotation": "confirmed_starter|expected_starter|no_mention|competition_lost|rotation_risk|ruled_out",
     "confidence": "high|medium|low"}
  ]
}

Judge every player you have an opinion about. Use no_mention only where you
genuinely have none. Do not explain."""


def memory_prompt(team: str, gw: int, date: str, opponent: str, roster: list[str]) -> str:
    return (f"TARGET CLUB: {team}\nFIXTURE: gameweek {gw} vs {opponent}, {date}\n"
            f"ROSTER ({len(roster)}): {', '.join(roster)}")


def run_memory(team, gw, date, opponent, roster) -> dict:
    raw = with_retry(_call_gemini, MEMORY_PROMPT,
                     memory_prompt(team, gw, date, opponent, roster))
    return raw


def signals_from_memory(data: dict, roster: list[str]) -> dict[str, float]:
    out = {}
    for p in (data.get("players") or []):
        if not isinstance(p, dict):
            continue
        n, r = str(p.get("name", "")), str(p.get("rotation", "no_mention")).lower()
        if n in roster and r in ROTATION_LABELS and ROTATION_LABELS[r] is not None:
            out[n] = ROTATION_LABELS[r]
    return out


def score_arm(rows: pd.DataFrame, col: str, label: str) -> dict:
    """How well does this arm's signal separate starters from non-starters?"""
    g = rows[rows[col].notna()]
    if g.empty:
        return {"label": label, "n": 0}
    base = rows.is_starter.mean()
    # rank-based: does a lower rotation value mean more likely to start?
    from sklearn.metrics import roc_auc_score
    auc = (roc_auc_score(g.is_starter, -g[col]) if g.is_starter.nunique() > 1 else np.nan)
    return {"label": label, "n": len(g), "coverage": len(g) / len(rows),
            "auc": auc, "base": base,
            "start_rate_flagged": g[g[col] >= 0.7].is_starter.mean() if (g[col] >= 0.7).any() else np.nan,
            "start_rate_backed": g[g[col] <= 0.2].is_starter.mean() if (g[col] <= 0.2).any() else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(DATA, encoding="utf-8-sig")
    # opponent and kickoff live in the source file; the modelling file keeps only
    # features, so merge the fixture metadata back in for the prompts.
    meta = (pd.read_csv(HERE / "dataset" / "pl_2025_2026_starting_xi.csv",
                        encoding="utf-8-sig")
              [["match_id", "team_code", "opponent_name", "kickoff_time"]]
              .drop_duplicates(["match_id", "team_code"]))
    df = df.merge(meta, on=["match_id", "team_code"], how="left")
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()

    targets = []
    for (mid, tc), grp in val.groupby(["match_id", "team_code"]):
        r = grp.iloc[0]
        targets.append({"team": r.team_name, "team_code": int(tc),
                        "gameweek": int(r.gameweek), "match_id": mid,
                        "date": str(r.kickoff_time)[:10],
                        "opponent": r.opponent_name, "roster": grp.player_name.tolist()})
    if args.limit:
        targets = targets[:args.limit]

    rule("LEAKAGE ABLATION", "#")
    print(f"  validation window GW{cut}+ | {len(targets)} team-matches")
    print(f"  Gemini calls needed: {2*len(targets)} (memory + shuffled), 0 Tavily credits")
    if not args.live:
        print("\n  --live not set: nothing executed.")
        return

    for d in CACHE.values():
        d.mkdir(exist_ok=True)

    # --- arm b: memory only ---
    rule("ARM B - MEMORY ONLY  (no sources supplied)")
    mem_rows = []
    for i, t in enumerate(targets, 1):
        p = CACHE["memory"] / f"gw{t['gameweek']:02d}_team{t['team_code']}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            try:
                data = run_memory(t["team"], t["gameweek"], t["date"],
                                  t["opponent"], t["roster"])
            except ExtractionFailure as e:
                data = {"players": [], "_error": str(e)}
            p.write_text(json.dumps(data), encoding="utf-8")
        sig = signals_from_memory(data, t["roster"])
        for n in t["roster"]:
            mem_rows.append({"match_id": t["match_id"], "team_code": t["team_code"],
                             "player_name": n, "mem_rotation": sig.get(n)})
        if i % 40 == 0:
            print(f"  {i}/{len(targets)}", flush=True)

    # --- arm c: shuffled sources ---
    rule("ARM C - SHUFFLED SOURCES  (another club's articles, same gameweek)")
    by_gw = {}
    for t in targets:
        by_gw.setdefault(t["gameweek"], []).append(t)
    rng = random.Random(0)
    shuf_rows = []
    done = 0
    for i, t in enumerate(targets, 1):
        pool = [x for x in by_gw[t["gameweek"]] if x["team_code"] != t["team_code"]]
        if not pool:
            continue
        other = rng.choice(pool)
        corpus = load_cached(other["team_code"], other["gameweek"], other["match_id"])
        if corpus is None or corpus.status != "ok" or not corpus.articles:
            continue
        # keep the TARGET roster, swap only the article text
        fake = Corpus(team=t["team"], team_code=t["team_code"], gameweek=t["gameweek"],
                      match_id=t["match_id"], match_date=corpus.match_date,
                      window_start=corpus.window_start, window_end=corpus.window_end,
                      articles=corpus.articles, status="ok")
        p = CACHE["shuffled"] / f"gw{t['gameweek']:02d}_team{t['team_code']}.json"
        if p.exists():
            ex = Extraction.from_dict(json.loads(p.read_text(encoding="utf-8")))
        else:
            from extraction import SYSTEM_PROMPT, build_user_prompt
            try:
                data = with_retry(_call_gemini, SYSTEM_PROMPT,
                                  build_user_prompt(t["roster"], fake))
                ex = verify(parse(data, fake), fake, roster=t["roster"])
            except ExtractionFailure as e:
                ex = Extraction(team_code=t["team_code"], gameweek=t["gameweek"],
                                match_id=t["match_id"], status="failed", error=str(e))
            p.write_text(json.dumps(ex.to_dict()), encoding="utf-8")
        got = {s.name: ROTATION_LABELS.get(s.rotation) for s in ex.players if s.verified}
        for n in t["roster"]:
            shuf_rows.append({"match_id": t["match_id"], "team_code": t["team_code"],
                              "player_name": n, "shuf_rotation": got.get(n)})
        done += 1
        if i % 40 == 0:
            print(f"  {i}/{len(targets)}", flush=True)
    print(f"  usable shuffled fixtures: {done}")

    # --- compare ---
    M = pd.DataFrame(mem_rows)
    S = pd.DataFrame(shuf_rows) if shuf_rows else pd.DataFrame(
        columns=["match_id", "team_code", "player_name", "shuf_rotation"])
    j = (val[["match_id", "team_code", "player_name", "is_starter",
              "llm_player_rotation"]]
         .merge(M, on=["match_id", "team_code", "player_name"], how="left")
         .merge(S, on=["match_id", "team_code", "player_name"], how="left"))
    j.to_csv(OUT / "leakage_ablation.csv", index=False, encoding="utf-8-sig")

    rule("RESULT")
    print(f"  {'arm':<34}{'n':>7}{'coverage':>10}{'AUC':>9}"
          f"{'flagged':>10}{'backed':>9}")
    print("  " + "-" * 80)
    for col, lab in [("llm_player_rotation", "A  sourced (verified quotes)"),
                     ("mem_rotation", "B  memory only, no sources"),
                     ("shuf_rotation", "C  shuffled sources")]:
        r = score_arm(j, col, lab)
        if not r.get("n"):
            print(f"  {lab:<34}{0:>7}")
            continue
        print(f"  {lab:<34}{r['n']:>7}{100*r['coverage']:>9.1f}%{r['auc']:>9.3f}"
              f"{r['start_rate_flagged']:>10.3f}{r['start_rate_backed']:>9.3f}")
    print(f"\n  base starter rate {j.is_starter.mean():.3f}")
    print("  AUC 0.5 = no information. 'flagged' = start rate among players the arm")
    print("  marked rotation_risk/ruled_out; 'backed' = marked confirmed/expected.")

    both = j[j.llm_player_rotation.notna() & j.mem_rotation.notna()]
    if len(both):
        agree = (both.llm_player_rotation == both.mem_rotation).mean()
        print(f"\n  where both A and B fire (n={len(both)}): they agree {100*agree:.1f}% "
              f"of the time")
    print(f"\n  saved -> reports/leakage_ablation.csv")


if __name__ == "__main__":
    main()
