"""Small live test of the retrieval + extraction stack before committing the full
season's budget.

Takes 5 team-matches from 5 different clubs at 5 well-separated gameweeks, then
retrieves the pre-match corpus (5 Tavily searches, 10 credits), extracts with
quote checking (5 Gemini calls), and scores what came out against what actually
happened. Did the players it called rotation risks get benched or not?

That last step is the whole point. You can eyeball retrieval quality yourself,
but only the outcome comparison tells you the signal carries anything.

python pilot.py            (uses cache; re-runnable at zero cost)
      python pilot.py --live     (permits network calls)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from extraction import ROTATION_LABELS, extract, to_rows
from retrieval import estimate_cost, fetch_team_context

HERE = Path(__file__).resolve().parent
FULL = HERE / "dataset" / "pl_2025_2026_starting_xi.csv"

# 5 clubs, 5 gameweeks spread across the season. Chosen for spacing and for a
# mix of contexts: an early-season week, two mid-season congested periods, a
# spring week, and a late-season week where objectives are settled.
# 15 distinct clubs across GW6-36. The first five are already cached from the
# earlier pilots, so only ten new searches are paid for.
PICKS = [("Liverpool", 8), ("Man City", 16), ("Arsenal", 23),
         ("Chelsea", 30), ("Newcastle", 36),
         ("Brentford", 6), ("Man Utd", 11), ("Wolves", 13), ("Spurs", 14),
         ("Aston Villa", 19), ("Brighton", 21), ("Everton", 25),
         ("West Ham", 27), ("Crystal Palace", 32), ("Fulham", 34)]


def rule(t: str, ch: str = "=") -> None:
    print("\n" + ch * 84)
    print(t)
    print(ch * 84)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="permit Tavily/Gemini calls (spends credits)")
    args = ap.parse_args()

    df = pd.read_csv(FULL, encoding="utf-8-sig")
    df["kick"] = pd.to_datetime(df.kickoff_time, errors="coerce", format="mixed", utc=True)

    # per-club fixture list, to find each match's predecessor for the window
    fixtures = (df.drop_duplicates(["team_code", "match_id"])
                  .sort_values("kick")[["team_code", "match_id", "kick", "gameweek"]])

    targets = []
    for team, gw in PICKS:
        sel = df[(df.team_name == team) & (df.gameweek == gw)]
        if sel.empty:
            print(f"  !! no rows for {team} GW{gw}")
            continue
        r = sel.iloc[0]
        tf = fixtures[fixtures.team_code == r.team_code]
        prev = tf[tf.kick < r.kick].tail(1)
        targets.append({
            "team": team, "team_code": int(r.team_code), "gameweek": int(gw),
            "match_id": r.match_id, "match_date": str(r.kick.date()),
            "prev_date": str(prev.kick.iloc[0].date()) if len(prev) else None,
            "opponent": r.opponent_name,
            "roster": sel.web_name.tolist(),
            "truth": dict(zip(sel.web_name, sel.is_starter)),
        })

    rule("PILOT PLAN", "#")
    c = estimate_cost(len(targets))
    print(f"  {len(targets)} team-matches | {c['credits']} Tavily credits "
          f"| {len(targets)} Gemini calls")
    print(f"  network: {'ENABLED' if args.live else 'OFF (cache only)'}\n")
    for t in targets:
        print(f"  {t['team']:<12} GW{t['gameweek']:<3} vs {str(t['opponent']):<14} "
              f"{t['match_date']}  prev {t['prev_date']}  squad {len(t['roster'])}")

    # --- 1. retrieval ---
    corpora = {}
    rule("1. RETRIEVAL")
    for t in targets:
        c = fetch_team_context(
            t["team"], t["team_code"], t["gameweek"], t["match_id"], t["match_date"],
            prev_match_date=t["prev_date"], opponent=t["opponent"],
            roster=t["roster"], allow_network=args.live)
        c.drop_excluded()          # apply any tightened exclusions to cached data
        corpora[t["team"]] = c
        print(f"\n  {t['team']} GW{t['gameweek']}  window {c.window_start} -> {c.window_end}"
              f"  status={c.status}  {c.n_kept}/{c.n_raw} kept")
        for a in c.articles:
            print(f"      [{a.score:>2}] {a.title[:78]}")
            print(f"           {a.url[:78]}")

    # --- 2. extraction ---
    rule("2. EXTRACTION  (quote-verified)")
    all_rows = []
    for t in targets:
        c = corpora[t["team"]]
        ex = extract(t["roster"], c, allow_network=args.live)
        print(f"\n  {t['team']} GW{t['gameweek']}  status={ex.status}  "
              f"verified={ex.n_verified} rejected={ex.n_rejected}")
        if ex.team.level != "none" or ex.team.verified:
            print(f"      TEAM ROTATION: {ex.team.level}  ({ex.team.reason})")
            print(f"        \"{ex.team.quote[:100]}\"")
        for p in ex.players:
            mark = "OK  " if p.verified else "NULL"
            print(f"      {mark} {p.name:<16} rot={p.rotation:<18} "
                  f"sent={p.sentiment:<12} {p.strength:<9} src={p.source_type}")
            if p.quote:
                print(f"           \"{p.quote[:96]}\"")
            if not p.verified:
                print(f"           rejected: {p.reject_reason}")
        rows = to_rows(ex, t["roster"])
        for r in rows:
            r["is_starter"] = t["truth"].get(r["player_name"])
            r["team_name"] = t["team"]
        all_rows.extend(rows)

    # --- 3. does it agree with reality? ---
    rule("3. VALIDITY  (against what actually happened)")
    R = pd.DataFrame(all_rows)
    R.to_csv(HERE / "reports" / "pilot_signals.csv", index=False, encoding="utf-8-sig")

    got = R[R.llm_player_rotation.notna()]
    print(f"  rows {len(R)} | players with a verified signal: {len(got)} "
          f"({100*len(got)/max(len(R),1):.1f}%)")
    print(f"  base starter rate in these squads: {R.is_starter.mean():.3f}")

    if len(got):
        print(f"\n  {'signal':<20}{'n':>5}{'started':>10}{'vs base':>10}")
        print("  " + "-" * 46)
        inv = {v: k for k, v in ROTATION_LABELS.items() if v is not None}
        for val, grp in got.groupby("llm_player_rotation"):
            print(f"  {inv.get(val, val):<20}{len(grp):>5}{grp.is_starter.mean():>10.3f}"
                  f"{grp.is_starter.mean()-R.is_starter.mean():>+10.3f}")

    tr = R[R.llm_team_rotation.notna()]
    if len(tr):
        print(f"\n  {'team rotation':<20}{'n':>5}{'started':>10}")
        print("  " + "-" * 36)
        for val, grp in tr.groupby("llm_team_rotation"):
            print(f"  {val:<20.2f}{len(grp):>5}{grp.is_starter.mean():>10.3f}")

    print(f"\n  saved -> reports/pilot_signals.csv")


if __name__ == "__main__":
    main()
