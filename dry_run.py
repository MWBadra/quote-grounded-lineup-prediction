"""Exercises the retrieval filter, the search window and the quote-verification
layer against hand-built fixtures. MAKES NO NETWORK CALLS and spends nothing.

Run this before authorising any live pilot: it proves the filtering separates
pre-match previews from match reports, that the window can never reach past
kickoff, and that an ungrounded quote is actually rejected rather than trusted.

python dry_run.py
"""

from __future__ import annotations

import sys

from extraction import (ROTATION_LABELS, Extraction, PlayerSignal, TeamSignal,
                        parse, quote_is_grounded, to_rows, verify)
from retrieval import (LOOKBACK_DAYS, Article, Corpus, filter_articles,
                       score_article, search_window, estimate_cost)

ROSTER = ["Salah", "Bradley", "Van Dijk", "Gakpo", "Szoboszlai",
          "Alisson", "Jones", "Chiesa", "Mac Allister"]

PASS, FAIL = "  [PASS]", "  [FAIL]"
failures = 0


def check(cond: bool, msg: str) -> None:
    global failures
    print((PASS if cond else FAIL) + " " + msg)
    if not cond:
        failures += 1


def rule(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# --- fixtures: what tavily realistically returns, good and bad ---
RAW_RESULTS = [
    # real pre-match previews. these have to survive
    {"title": "Liverpool team news: Conor Bradley out, Alisson a doubt for Spurs",
     "url": "https://liverpoolecho.co.uk/a", "published_date": "2026-03-12",
     "content": "Arne Slot confirmed that Conor Bradley will miss the rest of the season "
                "with a knee injury sustained in training. Alisson Becker is rated as a "
                "doubt and will be assessed on Friday. Van Dijk and Szoboszlai trained "
                "fully, while Gakpo is expected to keep his place."},
    {"title": "Predicted Liverpool XI to face Tottenham as Slot hints at changes",
     "url": "https://liverpoolecho.co.uk/b", "published_date": "2026-03-13",
     "content": "Slot suggested he is thinking about Wednesday's fixture and that some "
                "players need a rest, hinting at several changes to the side. Jones and "
                "Chiesa could come in, with Salah and Mac Allister rested."},
    {"title": "Arne Slot press conference: injury update and squad news",
     "url": "https://skysports.com/c", "published_date": "2026-03-13",
     "content": "The Liverpool boss said Mohamed Salah has been training well and will "
                "start on Saturday. He was less certain about Curtis Jones, and said "
                "Van Dijk is available after a knock."},
    # post-match reports. these have to be rejected
    {"title": "Liverpool 4-1 Barnsley: Szoboszlai scores stunner as Reds advance",
     "url": "https://skysports.com/d", "published_date": "2026-03-10",
     "content": "Dominik Szoboszlai scored a stunning goal as Liverpool beat the League "
                "One side. Salah and Gakpo also featured before being withdrawn."},
    {"title": "Liverpool player ratings: Salah superb in win over Barnsley",
     "url": "https://liverpoolecho.co.uk/e", "published_date": "2026-03-10",
     "content": "Mohamed Salah was the standout performer, rated 8 out of 10. Van Dijk "
                "and Bradley were solid."},
    {"title": "Liverpool vs Barnsley reaction: Slot delighted with response",
     "url": "https://bbc.com/f", "published_date": "2026-03-10",
     "content": "Slot said he was pleased with the reaction after full-time from Salah "
                "and Szoboszlai."},
    # wrong competition, reject
    {"title": "Sam Kerr becomes Chelsea's all-time top scorer in the WSL",
     "url": "https://espn.com/g", "published_date": "2026-03-11",
     "content": "The Australia striker reached the milestone on Sunday."},
]


def test_window() -> None:
    rule("1. SEARCH WINDOW  (v1 opened this on the previous match date)")
    s, e = search_window("2026-03-14")
    check(e < "2026-03-14", f"window ends before kickoff: {s} -> {e}")
    check(s == "2026-03-09", f"opens {LOOKBACK_DAYS} days out: {s}")

    s2, e2 = search_window("2026-03-14", prev_match_date="2026-03-11")
    check(s2 == "2026-03-12",
          f"clamped to the day AFTER the previous match: {s2} -> {e2} "
          "(so its reports fall outside)")

    s3, e3 = search_window("2026-03-14", prev_match_date="2026-03-13")
    check(s3 <= e3 < "2026-03-14",
          f"stays valid on a 1-day turnaround: {s3} -> {e3}")

    s4, e4 = search_window("2026-03-14", prev_match_date="2026-01-01")
    check(s4 == "2026-03-09",
          f"a distant previous match does not widen the window: {s4} -> {e4}")


def test_filter() -> None:
    rule("2. ARTICLE FILTER  (v1 discarded previews and kept reports)")
    arts = filter_articles(RAW_RESULTS, "Liverpool", roster=ROSTER)
    kept = [a.title for a in arts]
    print("  kept:")
    for a in arts:
        print(f"     score {a.score:>3}  {a.title[:66]}")
    dropped = [r["title"] for r in RAW_RESULTS if r["title"] not in kept]
    print("  dropped:")
    for t in dropped:
        print(f"                {t[:66]}")

    check(all("player ratings" not in t.lower() for t in kept), "player ratings dropped")
    check(all("4-1" not in t for t in kept), "scoreline report dropped")
    check(all("reaction" not in t.lower() for t in kept), "post-match reaction dropped")
    check(all("wsl" not in t.lower() for t in kept), "WSL story dropped")
    check(any("team news" in t.lower() for t in kept),
          "'team news' KEPT (v1's filter deleted exactly this)")
    check(any("predicted" in t.lower() for t in kept), "'predicted XI' kept")
    check(any("press conference" in t.lower() for t in kept), "press conference kept")
    check(len(arts) == 3, f"3 of 7 survive, all previews (got {len(arts)})")


def test_quote_grounding() -> None:
    rule("3. QUOTE GROUNDING  (the hallucination AND leakage control)")
    corpus = Corpus(team="Liverpool", team_code=14, gameweek=29,
                    match_id="m", match_date="2026-03-14",
                    window_start="2026-03-10", window_end="2026-03-13",
                    articles=filter_articles(RAW_RESULTS, "Liverpool", roster=ROSTER))
    text = corpus.searchable_text()

    check(quote_is_grounded(
        "Alisson Becker is rated as a doubt and will be assessed on Friday", text),
        "verbatim quote accepted")
    check(quote_is_grounded(
        "alisson becker IS RATED as a  doubt and will be assessed on Friday", text),
        "case and whitespace differences tolerated")
    check(not quote_is_grounded(
        "Slot confirmed Van Dijk will be rested against Tottenham", text),
        "INVENTED quote rejected (this is the leakage route)")
    check(not quote_is_grounded("a doubt", text),
          "too-short fragment rejected (would match by chance)")
    check(not quote_is_grounded("", text), "empty quote rejected")


def test_verification_pipeline() -> None:
    rule("4. END-TO-END VERIFY  (ungrounded signals must be nulled, not trusted)")
    corpus = Corpus(team="Liverpool", team_code=14, gameweek=29,
                    match_id="m", match_date="2026-03-14",
                    window_start="2026-03-10", window_end="2026-03-13",
                    articles=filter_articles(RAW_RESULTS, "Liverpool", roster=ROSTER))

    model_output = {
        "team_rotation": {
            "level": "high",
            "quote": "he is thinking about Wednesday's fixture and that some players need a rest",
            "source_index": 1, "reason": "prioritising midweek"},
        "players": [
            {"name": "Salah", "rotation": "confirmed_starter", "sentiment": "praised",
             "strength": "explicit", "source_index": 2,
             "quote": "Mohamed Salah has been training well and will start on Saturday"},
            {"name": "Bradley", "rotation": "ruled_out", "sentiment": "no_mention",
             "strength": "explicit", "source_index": 0,
             "quote": "Conor Bradley will miss the rest of the season"},
            # hallucinated. nothing in the sources says this
            {"name": "Van Dijk", "rotation": "rotation_risk", "sentiment": "no_mention",
             "strength": "explicit", "source_index": 1,
             "quote": "Van Dijk will be rested for this one"},
        ],
    }

    ex = verify(parse(model_output, corpus), corpus)
    v = {p.name: p.verified for p in ex.players}
    print(f"  verified {ex.n_verified}, rejected {ex.n_rejected}")
    for p in ex.players:
        mark = "OK  " if p.verified else "NULL"
        print(f"     {mark} {p.name:<10} {p.rotation:<18} {p.reject_reason}")

    check(v.get("Salah") is True, "grounded signal kept")
    check(v.get("Bradley") is True, "grounded signal kept")
    check(v.get("Van Dijk") is False, "HALLUCINATED signal rejected")
    check(ex.team.verified and ex.team.level == "high", "team rotation signal verified")

    roster = ["Salah", "Bradley", "Van Dijk", "Gakpo"]
    rows = {r["player_name"]: r for r in to_rows(ex, roster)}
    check(rows["Salah"]["llm_player_rotation"] == ROTATION_LABELS["confirmed_starter"],
          "Salah encoded as confirmed_starter (0.0)")
    check(rows["Bradley"]["llm_player_rotation"] == 1.0, "Bradley encoded as ruled_out (1.0)")
    check(rows["Van Dijk"]["llm_player_rotation"] is None,
          "rejected signal becomes NULL, not 0.0 (the v1 bug)")
    check(rows["Gakpo"]["llm_player_rotation"] is None, "unmentioned player is NULL")
    check(all(r["llm_team_rotation"] == 1.0 for r in rows.values()),
          "team rotation applies to every player in the squad")

    rule("5. TEAM WEIGHT IS SEPARATE, NOT MULTIPLIED")
    print("  Salah   player=0.0  team=1.0  -> two columns, model learns the interaction")
    print("  Gakpo   player=None team=1.0  -> team signal survives an absent player signal")
    check(rows["Gakpo"]["llm_team_rotation"] == 1.0 and
          rows["Gakpo"]["llm_player_rotation"] is None,
          "a squad-wide signal is not lost when a player is unmentioned")


def test_empty_corpus() -> None:
    rule("6. FAILURE STATES  (must be three, not two)")
    empty = Corpus(team="X", team_code=1, gameweek=5, match_id="m",
                   match_date="2026-01-01", window_start="a", window_end="b",
                   status="empty")
    ex = Extraction(team_code=1, gameweek=5, match_id="m", status="empty_corpus")
    rows = to_rows(ex, ["A", "B"])
    check(all(r["llm_player_rotation"] is None for r in rows),
          "empty corpus -> NULL for every player")
    check(rows[0]["llm_status"] == "empty_corpus",
          "status distinguishes empty corpus from a failed call")
    failed = Extraction(team_code=1, gameweek=5, match_id="m", status="failed")
    check(to_rows(failed, ["A"])[0]["llm_status"] == "failed",
          "failed call recorded as 'failed', never as a zero")


def test_budget() -> None:
    rule("7. BUDGET")
    for n, label in [(10, "pilot"), (740, "full season, GW2-38")]:
        c = estimate_cost(n)
        print(f"  {label:<24} {c['searches']:>4} searches  "
              f"{c['credits']:>5} credits  ~{c['minutes_at_100rpm']} min at 100 rpm")
    print("\n  Tavily budget with 2 keys: ~2,000 credits/month")
    print("  Full season at advanced depth: 1,480 credits (74%)")
    print("  Re-extraction after the fetch costs 0 credits -- corpus is cached on disk.")


if __name__ == "__main__":
    print("DRY RUN -- no network calls, no credits spent")
    test_window()
    test_filter()
    test_quote_grounding()
    test_verification_pipeline()
    test_empty_corpus()
    test_budget()
    print("\n" + "=" * 78)
    print(f"{'ALL CHECKS PASSED' if not failures else str(failures) + ' CHECK(S) FAILED'}")
    print("=" * 78)
    sys.exit(1 if failures else 0)
