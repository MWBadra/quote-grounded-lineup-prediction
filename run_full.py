"""Full-season LLM extraction for 2025-26, GW2-38 (740 team-matches).

Resumable by construction. Both stages cache to disk per (team, gameweek) and
that cache IS the checkpoint, so there's no separate progress file that can drift
out of step with what's actually on disk. Re-run and it skips whatever is done
already, so stopping for any reason only costs you the work in flight.

Running out of credit halts cleanly. A billing refusal raises CreditsExhausted,
which propagates rather than getting retried, and crucially is never cached - an
uncached fixture gets picked up on resume, whereas a cached empty one would be
skipped forever. The run stops, says where it got to, and tells you how to carry
on.

python run_full.py --stage retrieve     # Tavily only  (~1,480 credits)
      python run_full.py --stage extract      # Gemini only  (0 credits)
      python run_full.py --stage all
      python run_full.py --limit 50           # cap the number of NEW fixtures
      python run_full.py --dry                # plan only, no calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from extraction import cache_path as ex_cache
from extraction import extract, load_cached_extraction, to_rows
from extraction import is_cached as ex_cached
from llm_keys import CreditsExhausted, QuotaExhausted, gemini_pool, tavily_pool
from retrieval import (TAVILY_CREDITS_ADVANCED, fetch_team_context, load_cached)
from retrieval import is_cached as rt_cached

HERE = Path(__file__).resolve().parent
FULL = HERE / "dataset" / "pl_2025_2026_starting_xi.csv"
OUT = HERE / "dataset" / "llm_signals_2025_2026.csv"
QUERIES_PER_FIXTURE = 2


def rule(t: str, ch: str = "=") -> None:
    print("\n" + ch * 84)
    print(t)
    print(ch * 84, flush=True)


def hms(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def build_targets(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["kick"] = pd.to_datetime(df.kickoff_time, errors="coerce", format="mixed", utc=True)
    fixtures = (df.drop_duplicates(["team_code", "match_id"])
                  .sort_values("kick")[["team_code", "match_id", "kick"]])
    out = []
    for (mid, tcode), grp in df.groupby(["match_id", "team_code"], sort=False):
        r = grp.iloc[0]
        tf = fixtures[fixtures.team_code == tcode]
        prev = tf[tf.kick < r.kick].tail(1)
        out.append({
            "team": r.team_name, "team_code": int(tcode), "gameweek": int(r.gameweek),
            "match_id": mid, "match_date": str(r.kick.date()),
            "prev_date": str(prev.kick.iloc[0].date()) if len(prev) else None,
            "opponent": r.opponent_name, "roster": grp.web_name.tolist(),
            "kick": r.kick,
        })
    out.sort(key=lambda t: (t["gameweek"], t["team"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["retrieve", "extract", "all"], default="all")
    ap.add_argument("--limit", type=int, default=0, help="cap NEW fixtures this run")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(FULL, encoding="utf-8-sig")
    targets = build_targets(df)

    todo_r = [t for t in targets
              if not rt_cached(t["team_code"], t["gameweek"], t["match_id"])]
    todo_e = [t for t in targets
              if not ex_cached(t["team_code"], t["gameweek"], t["match_id"])]

    rule("PLAN", "#")
    print(f"  team-matches total        {len(targets)}")
    print(f"  retrieval already cached  {len(targets)-len(todo_r)}")
    print(f"  retrieval to fetch        {len(todo_r)}"
          f"   -> {len(todo_r)*QUERIES_PER_FIXTURE*TAVILY_CREDITS_ADVANCED:,} credits")
    print(f"  extraction already cached {len(targets)-len(todo_e)}")
    print(f"  extraction to run         {len(todo_e)}   -> {len(todo_e)} Gemini calls")
    try:
        print("\n  " + tavily_pool().report().replace("\n", "\n  "))
        print("  " + gemini_pool().report().splitlines()[0])
    except RuntimeError as e:
        print(f"  !! {e}")

    # Measured on the rev-3 pilot: ~2.6s per retrieval (two advanced searches)
    # and ~3.2s per extraction call, both dominated by provider latency rather
    # than by our rate limiter, which does not bind at these key counts.
    est = len(todo_r) * 2.6 + len(todo_e) * 3.2
    print(f"\n  estimated wall-clock      {hms(est)}"
          f"   ({hms(len(todo_r)*2.6)} retrieve + {hms(len(todo_e)*3.2)} extract)")
    if args.dry:
        print("\n  --dry: nothing executed.")
        return

    t0 = time.time()
    halted = None
    done_r = done_e = 0
    credits = 0

    # --- stage 1: retrieval ---
    if args.stage in ("retrieve", "all"):
        rule("RETRIEVAL")
        work = todo_r[:args.limit] if args.limit else todo_r
        for i, t in enumerate(work, 1):
            try:
                c = fetch_team_context(
                    t["team"], t["team_code"], t["gameweek"], t["match_id"],
                    t["match_date"], prev_match_date=t["prev_date"],
                    opponent=t["opponent"], roster=t["roster"], allow_network=True)
                credits += c.credits_spent
                done_r += 1
            except CreditsExhausted as e:
                halted = ("Tavily credits", str(e), t)
                break
            except QuotaExhausted as e:
                halted = ("Tavily daily quota", str(e), t)
                break
            if i % 20 == 0 or i == len(work):
                el = time.time() - t0
                rate = el / i
                print(f"  {i}/{len(work)}  {t['team']} GW{t['gameweek']}  "
                      f"{credits} credits  {hms(el)} elapsed  "
                      f"ETA {hms(rate*(len(work)-i))}", flush=True)

    # --- stage 2: extraction ---
    if args.stage in ("extract", "all") and halted is None:
        rule("EXTRACTION")
        todo_e = [t for t in targets
                  if not ex_cached(t["team_code"], t["gameweek"], t["match_id"])
                  and rt_cached(t["team_code"], t["gameweek"], t["match_id"])]
        work = todo_e[:args.limit] if args.limit else todo_e
        t1 = time.time()
        for i, t in enumerate(work, 1):
            corpus = load_cached(t["team_code"], t["gameweek"], t["match_id"])
            if corpus is None:
                continue
            corpus.drop_excluded()
            try:
                extract(t["roster"], corpus, allow_network=True)
                done_e += 1
            except CreditsExhausted as e:
                halted = ("Gemini credits", str(e), t)
                break
            except QuotaExhausted as e:
                halted = ("Gemini daily quota", str(e), t)
                break
            if i % 25 == 0 or i == len(work):
                el = time.time() - t1
                print(f"  {i}/{len(work)}  {t['team']} GW{t['gameweek']}  "
                      f"{hms(el)} elapsed  ETA {hms(el/i*(len(work)-i))}", flush=True)

    # --- assemble whatever exists ---
    rule("ASSEMBLING SIGNALS")
    rows = []
    for t in targets:
        ex = load_cached_extraction(t["team_code"], t["gameweek"], t["match_id"])
        if ex is None:
            continue
        for r in to_rows(ex, t["roster"]):
            # Stamp the TARGET fixture, not whatever the cached extraction carried.
            # Trusting ex.match_id is what produced 394 duplicated rows in the
            # signals file for the ten rescheduled-fixture collisions.
            r["match_id"] = t["match_id"]
            r["team_name"] = t["team"]
            rows.append(r)
    if rows:
        S = pd.DataFrame(rows)
        S.to_csv(OUT, index=False, encoding="utf-8-sig")
        got = S.llm_player_rotation.notna().sum()
        print(f"  {len(S):,} rows | {got:,} verified signals "
              f"({100*got/len(S):.1f}%) | {S.match_id.nunique()} matches")
        print(f"  saved -> {OUT}")
    else:
        print("  nothing extracted yet")

    # --- outcome ---
    rule("RESULT")
    print(f"  retrieved this run {done_r} | extracted this run {done_e} "
          f"| credits spent {credits}")
    print(f"  wall-clock {hms(time.time()-t0)}")
    if halted:
        what, msg, t = halted
        rule("HALTED - " + what, "!")
        print(f"  stopped at: {t['team']} GW{t['gameweek']} ({t['match_date']})")
        print(f"  provider said: {msg[:200]}")
        print("\n  Nothing is lost. The cache is the checkpoint, and the fixture that")
        print("  failed was NOT cached, so it will be retried first on resume.")
        print("\n  To continue:")
        print("    1. top up or add keys in .env (TAVILY_API_KEY_3, GEMINI_..._KEY_9, ...)")
        print("       -- any env var starting with TAVILY / GEMINI is picked up")
        print("    2. re-run the exact same command; it resumes where it stopped")
        remaining = len([x for x in targets
                         if not rt_cached(x["team_code"], x["gameweek"], x["match_id"])])
        print(f"\n  remaining to retrieve: {remaining} fixtures "
              f"(~{remaining*QUERIES_PER_FIXTURE*TAVILY_CREDITS_ADVANCED:,} credits, "
              f"~{hms(remaining*2.6)})")
        sys.exit(2)
    print("  completed without hitting a limit")


if __name__ == "__main__":
    main()
