"""Version 2, step 1: build the clean tabular dataset for starting-XI prediction.

One row per (player, team-match), for every player in that club's availability
pool that gameweek. 2025-26 Premier League only, GW2 to GW38. GW1 is dropped
because there's no GW0 snapshot to read pre-match state from, so the only thing
we could build a GW1 row out of is GW1's own post-match stats.

Leakage rules, enforced everywhere below:
  - season stats and availability come off the GW(N-1) snapshot
  - selection history and rolling form only use matches strictly before N
  - position is the modal slot over PRIOR starts, never including this one
  - average_positions for match N builds match N's label and nothing else.
    it is never a feature for the match it came from

python build_dataset.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_dataset import fetch_csv, get_repo_meta, get_tree, build_inventory  # noqa: E402

SEASON = "2025-2026"
COMP = "Premier League"
FIRST_GW, LAST_GW = 2, 38

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "dataset"

# Season-cumulative columns lifted straight from the GW(N-1) playerstats snapshot.
# Identical for every outfield position on purpose: an attacker may be asked to
# play midfield and a midfielder to drop into defence, so the model needs the
# same evidence for everyone rather than a position-conditional subset.
SEASON_STATS = [
    # attacking
    "goals_scored", "assists", "expected_goals", "expected_assists",
    "expected_goal_involvements",
    # defending. populated for every position, not just defenders
    "clean_sheets", "goals_conceded", "expected_goals_conceded",
    "defensive_contribution", "tackles", "clearances_blocks_interceptions",
    "recoveries",
    # discipline
    "yellow_cards", "red_cards",
    # standing / involvement
    "minutes", "starts", "form", "points_per_game", "total_points",
    "bonus", "bps", "ict_index", "influence", "creativity", "threat",
    "selected_by_percent", "now_cost",
    # set pieces. `set_piece_threat` is dropped: the upstream column is 100%
    # empty for this season. The three *_order columns are null for players who
    # are not designated takers, which is information rather than missingness,
    # so they are filled with 0 and summarised by is_set_piece_taker below.
    "penalties_order",
    "corners_and_indirect_freekicks_order", "direct_freekicks_order",
]

SET_PIECE_ORDER = ["penalties_order", "corners_and_indirect_freekicks_order",
                   "direct_freekicks_order"]

# Goalkeeper-only. Structurally zero for outfield players, kept separate so the
# model is not fed a block that is constant-zero for 90% of rows.
GK_STATS = ["saves", "penalties_saved", "saves_per_90", "clean_sheets_per_90",
            "goals_conceded_per_90"]

# Per-match columns rolled into last-5 means. Same block for everyone.
FORM_STATS = [
    "minutes_played",
    "accurate_passes", "accurate_passes_percent", "final_third_passes",
    "chances_created",
    "xg", "xa", "total_shots", "shots_on_target", "touches_opposition_box",
    "successful_dribbles",
    "tackles_won", "interceptions", "clearances", "blocks",
    "aerial_duels_won", "duels_won", "dribbled_past", "recoveries",
    "saves", "goals_conceded", "goals_prevented", "xgot_faced",
    "distance_covered", "sprinting_distance", "number_of_sprints", "top_speed",
    "fouls_committed", "was_fouled",
]

BROAD_TO_POSITION = {"G": "GK", "D": "D-CENTRE", "M": "M-CENTRE", "F": "F-CENTRE"}
FPL_TO_BROAD = {"Goalkeeper": "G", "Defender": "D", "Midfielder": "M", "Forward": "F"}


# --- loading ---

def load_inventory() -> pd.DataFrame:
    inv_path = HERE / "reports" / "inventory.csv"
    if inv_path.exists():
        return pd.read_csv(inv_path)
    meta = get_repo_meta()
    return build_inventory(get_tree(meta["default_branch"]))


def paths_for(inv: pd.DataFrame, table: str, competition: str | None) -> dict[int, str]:
    """gameweek -> path, for one table. competition=None means season-level files."""
    sel = inv[(inv.season == SEASON) & (inv.table == table)]
    if competition:
        sel = sel[sel.competition == competition]
    sel = sel[sel.gameweek.notna()]
    return {int(r.gameweek): r.path for _, r in sel.iterrows()}


def load_gw_table(inv: pd.DataFrame, table: str, competition: str | None,
                  tag_gw: bool = True) -> pd.DataFrame:
    """Concatenate every gameweek file for a table, tagging the source gameweek."""
    frames = []
    for gw, path in sorted(paths_for(inv, table, competition).items()):
        df = fetch_csv("main", path)
        if df is None or df.empty:
            continue
        if tag_gw:
            df = df.assign(_gw=gw)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- slot derivation ---

def assign_slots(group: pd.DataFrame) -> pd.Series:
    """
    Within one positional line of a team-match, decide WIDE vs CENTRE.

    The two outermost players of a line are WIDE, but only when the line has at
    least three players and they actually sit off-centre. That keeps a double
    pivot central, keeps two strikers central, and correctly marks the outer
    two of a back three or a front three as wide.
    """
    n = len(group)
    out = pd.Series("CENTRE", index=group.index)
    if n < 3:
        return out
    order = group["y"].sort_values()
    for idx in (order.index[0], order.index[-1]):
        if abs(group.loc[idx, "y"] - 50.0) > 12.0:
            out.loc[idx] = "WIDE"
    return out


def build_slot_labels(avgpos: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    """
    (match_id, player_id) -> WIDE/CENTRE position label.

    Starters are labelled by rank within their positional line, which needs a
    complete line to be meaningful. Substitutes are labelled by absolute width
    instead: they share the pitch with a partial line, so their rank within it
    would be arbitrary. Including substitutes matters because a squad player who
    only ever comes off the bench would otherwise never acquire a real position
    and would sit at the broad-position fallback all season.
    """
    a = avgpos.dropna(subset=["x", "y", "position"]).copy()
    key = set(zip(starters.match_id, starters.player_id))
    a["_is_start"] = [(m, p) in key for m, p in zip(a.match_id, a.player_id)]

    st = a[a._is_start]
    parts = [assign_slots(line) for _, line in
             st.groupby(["match_id", "team_side", "position"], sort=False)]
    st = st.assign(_side=pd.concat(parts) if parts else "CENTRE")

    sub = a[~a._is_start].copy()
    sub["_side"] = np.where((sub["y"] - 50.0).abs() > 18.0, "WIDE", "CENTRE")

    s = pd.concat([st, sub], ignore_index=True)
    s["slot_position"] = np.where(
        s["position"] == "G", "GK", s["position"] + "-" + s["_side"]
    )
    return s[["match_id", "player_id", "slot_position"]]


# --- main build ---

def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inv = load_inventory()

    print("=" * 78)
    print("LOADING")
    print("=" * 78)

    matches = load_gw_table(inv, "matches", COMP).drop_duplicates("match_id")
    matches["kickoff_time"] = pd.to_datetime(matches["kickoff_time"],
                                             errors="coerce", format="mixed", utc=True)
    print(f"  matches            {len(matches):>6}")

    lineups = load_gw_table(inv, "lineups", COMP).drop_duplicates(
        ["match_id", "team_side", "jersey_number", "player_name"])
    lineups = lineups[lineups.player_id.notna()].copy()
    lineups["player_id"] = lineups["player_id"].astype(int)
    print(f"  lineup rows        {len(lineups):>6}")

    avgpos = load_gw_table(inv, "average_positions", COMP)
    avgpos = avgpos[avgpos.player_id.notna()].copy()
    avgpos["player_id"] = avgpos["player_id"].astype(int)
    avgpos = avgpos.drop_duplicates(["match_id", "player_id"])
    print(f"  average_positions  {len(avgpos):>6}")

    pms = load_gw_table(inv, "playermatchstats", COMP).drop_duplicates(
        ["match_id", "player_id"])
    print(f"  playermatchstats   {len(pms):>6}")

    pstats = load_gw_table(inv, "playerstats", None)
    players = load_gw_table(inv, "players", None)
    print(f"  playerstats snaps  {len(pstats):>6}   players snaps {len(players):>6}")

    teams = fetch_csv("main", f"data/{SEASON}/teams.csv")
    code2name = dict(zip(teams["code"], teams["name"]))
    # NOTE: matches.home_team / away_team hold team CODES, not teams.id. The two
    # ranges overlap (ids are 1..20, codes include 1,2,3,4,6,7,8,11,14,17), so
    # mapping through teams.id silently yields the wrong club for most rows
    # instead of failing loudly. Use the value directly.

    # --- labels ---
    starters = lineups[lineups.is_starting].copy()
    slot_lab = build_slot_labels(avgpos, starters)
    print(f"  slot labels        {len(slot_lab):>6}")

    # --- chronology ---
    matches = matches.sort_values("kickoff_time")
    mdate = dict(zip(matches.match_id, matches.kickoff_time))
    mgw = dict(zip(matches.match_id, matches.gameweek))

    lineups["kickoff"] = lineups.match_id.map(mdate)
    lineups["gameweek"] = lineups.match_id.map(mgw)
    lineups["team_code"] = lineups["team_code"].astype("Int64")

    # team fixture calendar, Premier League only
    team_fixtures: dict[int, list] = defaultdict(list)
    for _, r in lineups.drop_duplicates(["match_id", "team_code"]).iterrows():
        if pd.notna(r.team_code):
            team_fixtures[int(r.team_code)].append((r.kickoff, r.match_id))
    for tc in team_fixtures:
        team_fixtures[tc].sort()

    # opponent lookup
    opp = {}
    for mid, grp in lineups.drop_duplicates(["match_id", "team_code"]).groupby("match_id"):
        codes = [int(c) for c in grp.team_code.dropna().unique()]
        if len(codes) == 2:
            opp[(mid, codes[0])] = codes[1]
            opp[(mid, codes[1])] = codes[0]

    # --- per-player chronological timeline ---
    # Built from LINEUPS, not playermatchstats: lineups is the authoritative
    # record of squad membership, so a player who was benched still gets a
    # timeline entry. Keying history off playermatchstats would silently give
    # zero history to every pool member who did not get on the pitch.
    form_cols = [c for c in FORM_STATS if c in pms.columns]
    stat_lut = pms.set_index(["match_id", "player_id"])[form_cols]

    tl = lineups[["match_id", "player_id", "team_code", "is_starting", "kickoff"]].copy()
    tl = tl.merge(slot_lab, on=["match_id", "player_id"], how="left")
    tl = tl.merge(pms[["match_id", "player_id"] + form_cols],
                  on=["match_id", "player_id"], how="left")
    tl = tl.sort_values(["player_id", "kickoff"])

    timeline: dict[int, dict] = {}
    for pid, d in tl.groupby("player_id", sort=False):
        timeline[pid] = {
            "dates": d["kickoff"].tolist(),
            "started": d["is_starting"].astype(bool).tolist(),
            "slots": d["slot_position"].tolist(),
            "teams": d["team_code"].tolist(),
            "minutes": d["minutes_played"].fillna(0).tolist()
            if "minutes_played" in d.columns else [0] * len(d),
            "stats": d[form_cols].to_numpy(dtype=float),
        }

    def history_before(pid: int, when, tcode: int) -> tuple[dict, str | None, np.ndarray | None]:
        """Everything known about a player strictly before `when`."""
        t = timeline.get(pid)
        if t is None:
            return _EMPTY_HIST.copy(), None, None
        # dates are sorted; find the cut point
        i = np.searchsorted(np.array(t["dates"]), when, side="left")
        if i == 0:
            return _EMPTY_HIST.copy(), None, None

        st = t["started"][:i]
        dates = t["dates"][:i]
        mins = t["minutes"][:i]
        slots = [s for s in t["slots"][:i] if isinstance(s, str)]

        run = 0
        for v in reversed(st):
            if v:
                run += 1
            else:
                break
        last_start = next((dates[j] for j in range(i - 1, -1, -1) if st[j]), None)

        h = dict(
            started_last_1=int(st[-1]),
            started_last_3=int(sum(st[-3:])),
            started_last_5=int(sum(st[-5:])),
            start_rate_last_5=float(np.mean(st[-5:])),
            starts_season=int(sum(st)),
            minutes_season=float(np.nansum(mins)),
            consecutive_starts=run,
            days_since_last_start=((when - last_start).days if last_start is not None else -1),
            was_unused_sub_last_match=int((not st[-1]) and (mins[-1] == 0)),
            n_prior_apps=i,
            n_prior_apps_this_club=int(sum(1 for c in t["teams"][:i] if c == tcode)),
        )
        modal = pd.Series(slots).mode().iloc[0] if slots else None
        window = t["stats"][max(0, i - 5):i]
        form = np.nanmean(window, axis=0) if len(window) else None
        return h, modal, form

    _EMPTY_HIST = dict(
        started_last_1=0, started_last_3=0, started_last_5=0, start_rate_last_5=0.0,
        starts_season=0, minutes_season=0.0, consecutive_starts=0,
        days_since_last_start=-1, was_unused_sub_last_match=0,
        n_prior_apps=0, n_prior_apps_this_club=0,
    )
    form_names = [f"f5_{c}" for c in form_cols]

    # --- assemble ---
    print("\n" + "=" * 78)
    print("ASSEMBLING")
    print("=" * 78)

    snap_stats = {gw: d for gw, d in pstats.groupby("_gw")}
    snap_players = {gw: d for gw, d in players.groupby("_gw")}

    matchday = set(zip(lineups.match_id, lineups.player_id))
    start_lut = dict(
        ((m, p), 1) for m, p in zip(starters.match_id, starters.player_id))
    formation_lut = dict(
        ((m, int(t)), f)
        for m, t, f in zip(starters.match_id, starters.team_code, starters.formation)
        if pd.notna(t))
    minutes_lut = dict(
        ((m, p), v) for m, p, v in zip(pms.match_id, pms.player_id, pms.minutes_played))

    rows = []
    for gw in range(FIRST_GW, LAST_GW + 1):
        prev = gw - 1
        if prev not in snap_stats or prev not in snap_players:
            continue
        ps = snap_stats[prev]
        pl = snap_players[prev][["player_id", "player_code", "team_code", "position"]]
        snap = ps.merge(pl, left_on="id", right_on="player_id", how="inner")
        # availability pool: available or doubtful at the previous snapshot
        snap = snap[snap.status.isin(["a", "d"])]

        gw_matches = matches[matches.gameweek == gw]
        for _, mt in gw_matches.iterrows():
            mid = mt.match_id
            for side in ("home", "away"):
                raw = mt["home_team"] if side == "home" else mt["away_team"]
                if pd.isna(raw):
                    continue
                tcode = int(raw)  # already a team CODE, see note above
                pool = snap[snap.team_code == tcode]
                if pool.empty:
                    continue
                ocode = opp.get((mid, int(tcode)))
                orow = teams[teams["code"] == ocode]
                fixtures = team_fixtures.get(int(tcode), [])
                idx = next((i for i, (_, m) in enumerate(fixtures) if m == mid), None)
                prev_ko = fixtures[idx - 1][0] if idx and idx > 0 else None
                next_ko = fixtures[idx + 1][0] if idx is not None and idx + 1 < len(fixtures) else None

                for _, p in pool.iterrows():
                    pid = int(p.player_id)
                    h, modal, form = history_before(pid, mt.kickoff_time, tcode)

                    broad = FPL_TO_BROAD.get(p.position, "M")
                    inferred = modal is None
                    pos = BROAD_TO_POSITION[broad] if inferred else modal

                    row = {
                        # --- identifiers (metadata, never features) ---
                        "season": SEASON,
                        "gameweek": gw,
                        "match_id": mid,
                        "kickoff_time": mt.kickoff_time,
                        "team_code": int(tcode),
                        "team_name": code2name.get(tcode),
                        "opponent_code": ocode,
                        "opponent_name": code2name.get(ocode),
                        "is_home": int(side == "home"),
                        "player_id": pid,
                        "player_code": p.get("player_code"),
                        "web_name": p.get("web_name"),
                        "fpl_position": p.position,
                        "status": p.status,
                        # --- labels ---
                        "is_starter": int(start_lut.get((mid, pid), 0)),
                        "formation": formation_lut.get((mid, int(tcode))),
                        "in_matchday_squad": int((mid, pid) in matchday),
                        "minutes_played": minutes_lut.get((mid, pid), 0),
                        # --- position (single column, as requested) ---
                        "player_position": pos,
                        "position_is_inferred": int(inferred),
                        # --- availability ---
                        # Read from the GW(N-1) snapshot. The GW(N) snapshot is
                        # written after the match, so it stamps `i` on players who
                        # were injured during the fixture we're predicting, so using
                        # it drops 5.2% of genuine starters from the pool and is
                        # less correlated with starting, not more.
                        #
                        # availability_score grades the doubt: `chance_of_playing
                        # _next_round` is FPL's own forward-looking estimate for
                        # exactly this gameweek, and for doubtful players it
                        # resolves to 25/50/75 rather than a flat "maybe".
                        "is_available": int(p.status == "a"),
                        "availability_score": (
                            100.0 if p.status == "a"
                            else pd.to_numeric(p.get("chance_of_playing_next_round"),
                                               errors="coerce")),
                        # --- rest ---
                        "days_since_prev_match": (
                            (mt.kickoff_time - prev_ko).days if prev_ko is not None else -1),
                        "days_until_next_match": (
                            (next_ko - mt.kickoff_time).days if next_ko is not None else -1),
                        # --- transfer flags ---
                        # new signing: never appeared for THIS club before.
                        # new to league: no prior Premier League appearance at all.
                        "is_new_signing": int(h["n_prior_apps_this_club"] == 0),
                        "is_new_to_league": int(h["n_prior_apps"] == 0),
                        "has_history": int(h["n_prior_apps"] > 0),
                        # --- opponent strength only ---
                        "opp_elo": float(orow["elo"].iloc[0]) if len(orow) and pd.notna(orow["elo"].iloc[0]) else np.nan,
                        "opp_strength_overall": (
                            float(orow["strength_overall_away"].iloc[0]) if side == "home"
                            else float(orow["strength_overall_home"].iloc[0])) if len(orow) else np.nan,
                        "opp_strength_attack": (
                            float(orow["strength_attack_away"].iloc[0]) if side == "home"
                            else float(orow["strength_attack_home"].iloc[0])) if len(orow) else np.nan,
                        "opp_strength_defence": (
                            float(orow["strength_defence_away"].iloc[0]) if side == "home"
                            else float(orow["strength_defence_home"].iloc[0])) if len(orow) else np.nan,
                    }
                    row.update(h)
                    row.pop("n_prior_apps")
                    row.pop("n_prior_apps_this_club")

                    for c in SEASON_STATS:
                        row[c] = p.get(c, np.nan)
                    yc = pd.to_numeric(p.get("yellow_cards"), errors="coerce")
                    row["yellows_from_ban"] = (
                        int(5 - yc % 5) if pd.notna(yc) and yc < 10 else -1)
                    for c in GK_STATS:
                        row[f"gk_{c}"] = p.get(c, np.nan) if broad == "G" else 0.0

                    if form is not None:
                        row.update(dict(zip(form_names, form)))
                    else:
                        for c in form_names:
                            row[c] = np.nan
                    rows.append(row)

        if gw % 6 == 0:
            print(f"  ... through GW{gw}: {len(rows):,} rows")

    df = pd.DataFrame(rows)

    # Set-piece duty: null upstream means "not a designated taker", so encode
    # that explicitly rather than leaving it as missing data.
    for c in SET_PIECE_ORDER:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["is_set_piece_taker"] = (df[SET_PIECE_ORDER].gt(0).any(axis=1)).astype(int)

    # --- report ---
    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  rows              {len(df):,}")
    print(f"  columns           {df.shape[1]}")
    print(f"  team-matches      {df.groupby(['match_id','team_code']).ngroups}")
    print(f"  gameweeks         {df.gameweek.min()}..{df.gameweek.max()}")
    print(f"  pool size         mean {df.groupby(['match_id','team_code']).size().mean():.1f}"
          f"  min {df.groupby(['match_id','team_code']).size().min()}"
          f"  max {df.groupby(['match_id','team_code']).size().max()}")
    print(f"  starter rate      {df.is_starter.mean():.4f}")

    per = df.groupby(["match_id", "team_code"]).is_starter.sum()
    print(f"  starters per team-match: {per.value_counts().sort_index().to_dict()}")

    print(f"\n  player_position:  {df.player_position.value_counts().to_dict()}")
    print(f"  inferred position: {100*df.position_is_inferred.mean():.1f}% of rows")
    print(f"  formation coverage: {100*df.formation.notna().mean():.1f}%")

    out = OUT_DIR / "pl_2025_2026_starting_xi.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    pd.DataFrame({"column": df.columns,
                  "non_null_pct": (100 * df.notna().mean()).round(1).values,
                  "dtype": [str(t) for t in df.dtypes]}
                 ).to_csv(OUT_DIR / "columns.csv", index=False)
    print(f"\n  saved -> {out}")
    print(f"  saved -> {OUT_DIR / 'columns.csv'}")
    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
