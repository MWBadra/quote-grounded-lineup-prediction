"""Version 2, step 0: find out what is actually in the upstream dataset before
building anything on top of it.

v1 hardcoded four raw.githubusercontent URLs pointing at olbauday/FPL-Elo-Insights
and assumed the flat 2024-25 layout. Neither holds any more: the repo has been
renamed to FPL-Core-Insights, it carries three seasons now, and the newer ones
are partitioned per gameweek and per competition with tables v1 never saw at all
(lineups.csv, incidents.csv, average_positions.csv and others).

So nothing here is hardcoded. Inventory the repo off the GitHub API first, then
profile the schema of whatever tables turn up.

    python explore_dataset.py                  # inventory + schema profile
    python explore_dataset.py --deep           # concat every gameweek of the key
                                               # tables for real row counts,
                                               # date ranges, coverage
    python explore_dataset.py --season 2025-2026
    python explore_dataset.py --no-cache       # skip the local file cache

On rate limits: the whole tree comes back in one call (git/trees?recursive=1)
and unauthenticated GitHub gives you 60/hour, so this is cheap. CSVs come from
raw.githubusercontent.com which isn't limited the same way. Set GITHUB_TOKEN if
you ever need the 5000/hour ceiling.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd

# --- upstream identity ---
# Address the repo by NUMERIC ID, not by owner/name. GitHub keeps the numeric id
# stable across renames, which is exactly the failure that broke v1's URLs when
# FPL-Elo-Insights became FPL-Core-Insights.
REPO_ID = 909878662
API_ROOT = f"https://api.github.com/repositories/{REPO_ID}"

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / ".cache"
REPORT_DIR = HERE / "reports"

# Tables worth profiling in --deep mode (concatenated across all gameweeks).
# These are the ones that matter for starting-XI prediction.
KEY_TABLES = ["lineups", "matches", "fixtures", "playermatchstats", "players", "teams"]

# Columns that identify a row's time position, used to report date coverage.
DATE_COLUMN_CANDIDATES = ["kickoff_time", "match_date", "date", "kickoff", "deadline_time"]


# --- http helpers ---

def _request(url: str, accept: str = "application/vnd.github+json") -> bytes:
    """GET with a User-Agent (GitHub rejects requests without one) and optional token."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "fpl-v2-explorer")
    req.add_header("Accept", accept)
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def api_json(path: str) -> dict | list:
    """Call the GitHub REST API and parse JSON."""
    url = path if path.startswith("http") else f"{API_ROOT}{path}"
    try:
        return json.loads(_request(url))
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise SystemExit(
                "GitHub API rate limit hit (60/hour unauthenticated).\n"
                "Set GITHUB_TOKEN in your environment to raise it to 5000/hour, "
                "or wait an hour. Cached files in .cache/ still work with --no-network."
            ) from e
        raise


def raw_url(branch: str, path: str) -> str:
    """Build a raw.githubusercontent URL, URL-encoding the spaces in 'By Gameweek' etc."""
    quoted = urllib.parse.quote(path)
    return f"https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/{branch}/{quoted}"


def fetch_csv(branch: str, path: str, use_cache: bool = True) -> pd.DataFrame | None:
    """Download one CSV into a DataFrame, caching the bytes on disk."""
    cache_path = CACHE_DIR / path.replace("/", "__")
    if use_cache and cache_path.exists():
        data = cache_path.read_bytes()
    else:
        try:
            data = _request(raw_url(branch, path), accept="text/plain")
        except urllib.error.HTTPError as e:
            print(f"    ! could not fetch {path}: HTTP {e.code}")
            return None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    try:
        # utf-8-sig: the upstream files carry a BOM, which is what corrupted
        # names like "Nørgaard" in v1 until it was patched downstream.
        return pd.read_csv(io.BytesIO(data), encoding="utf-8-sig", low_memory=False)
    except Exception as e:  # noqa: BLE001 - report and continue, one bad file shouldn't stop the survey
        print(f"    ! could not parse {path}: {e}")
        return None


# --- inventory ---

def get_repo_meta() -> dict:
    d = api_json("")
    return {
        "full_name": d["full_name"],
        "default_branch": d["default_branch"],
        "size_kb": d["size"],
        "pushed_at": d["pushed_at"],
        "description": d.get("description", ""),
    }


def get_tree(branch: str) -> list[dict]:
    """Entire repo file listing in a single API call."""
    d = api_json(f"/git/trees/{branch}?recursive=1")
    if d.get("truncated"):
        print("  ! WARNING: tree response was truncated by GitHub; inventory is partial.")
    return [n for n in d["tree"] if n["type"] == "blob"]


def parse_path(path: str) -> dict | None:
    """
    Turn a repo path into structured fields.

    Handles both layouts:
      data/2024-2025/matches/GW1/matches.csv          -> old, per-GW under table
      data/2024-2025/matches/matches.csv              -> old, season-level
      data/2025-2026/By Gameweek/GW1/lineups.csv      -> new, per-GW
      data/2025-2026/By Tournament/Europa League/GW3/matches.csv
      data/2025-2026/players.csv                      -> new, season-level
    """
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "data" or not path.endswith(".csv"):
        return None

    season = parts[1]
    table = Path(parts[-1]).stem
    mid = parts[2:-1]  # everything between the season and the filename

    scope, competition, gameweek = "season", None, None
    for seg in mid:
        if seg.startswith("GW") and seg[2:].isdigit():
            scope, gameweek = "gameweek", int(seg[2:])
        elif seg == "By Gameweek":
            competition = "All"
        elif seg == "By Tournament":
            competition = competition or "?"
        elif seg not in (table,):
            # a competition name, or the old layout's table folder
            if competition == "?" or competition is None:
                competition = seg

    return {
        "path": path,
        "season": season,
        "competition": competition or "All",
        "gameweek": gameweek,
        "scope": scope,
        "table": table,
    }


def build_inventory(tree: list[dict]) -> pd.DataFrame:
    rows = []
    for node in tree:
        parsed = parse_path(node["path"])
        if parsed:
            parsed["size"] = node.get("size", 0)
            rows.append(parsed)
    return pd.DataFrame(rows)


# --- reporting ---

def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def rule(title: str, char: str = "=") -> None:
    print("\n" + char * 78)
    print(title)
    print(char * 78)


def report_seasons(inv: pd.DataFrame) -> None:
    rule("SEASONS AVAILABLE")
    g = inv.groupby("season").agg(
        files=("path", "size"), tables=("table", "nunique"), bytes=("size", "sum")
    )
    print(f"{'season':<12}{'csv files':>11}{'distinct tables':>18}{'total size':>13}")
    print("-" * 78)
    for season, r in g.iterrows():
        print(f"{season:<12}{r['files']:>11,}{r['tables']:>18}{human(r['bytes']):>13}")
    print(f"\n  v1 used ONLY 2024-2025. {len(g)} seasons are now available.")


def report_layout(inv: pd.DataFrame) -> None:
    rule("LAYOUT PER SEASON")
    for season in sorted(inv["season"].unique()):
        s = inv[inv["season"] == season]
        comps = sorted(s["competition"].unique())
        gws = s[s["gameweek"].notna()]["gameweek"]
        print(f"\n  {season}")
        print(f"    competitions : {', '.join(comps)}")
        if len(gws):
            print(f"    gameweeks    : GW{int(gws.min())} .. GW{int(gws.max())} "
                  f"({gws.nunique()} distinct)")
        season_lvl = sorted(s[s["scope"] == "season"]["table"].unique())
        if season_lvl:
            print(f"    season-level : {', '.join(season_lvl)}")


def report_table_matrix(inv: pd.DataFrame) -> None:
    rule("TABLE x SEASON COVERAGE  (number of csv files)")
    pivot = inv.pivot_table(
        index="table", columns="season", values="path", aggfunc="count", fill_value=0
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOTAL", ascending=False)
    print(pivot.to_string())
    print("\n  Tables v1 never touched are the interesting ones — notably `lineups`")
    print("  (real starting XIs), `incidents`, and `average_positions`.")


def report_competitions(inv: pd.DataFrame) -> None:
    rule("COMPETITION COVERAGE  (fixture-congestion signal)")
    sub = inv[inv["competition"] != "All"]
    if sub.empty:
        print("  none found")
        return
    g = sub.groupby(["season", "competition"]).agg(
        files=("path", "size"), gameweeks=("gameweek", "nunique")
    )
    print(g.to_string())
    print("\n  Midweek European and cup fixtures are the actual mechanism behind")
    print("  squad rotation. v1 had no visibility into them at all.")


def profile_schema(branch: str, inv: pd.DataFrame, use_cache: bool, season: str | None) -> dict:
    """Download one representative file per (season, table) and report its schema."""
    rule("SCHEMA PROFILE  (one representative file per table)")
    out = {}
    target = inv if season is None else inv[inv["season"] == season]

    # Take the largest file per (season, table). Most likely to be a full
    # season-level dump rather than a sparse single gameweek.
    picks = (target.sort_values("size", ascending=False)
                   .groupby(["season", "table"], as_index=False)
                   .first())

    for _, row in picks.sort_values(["season", "table"]).iterrows():
        df = fetch_csv(branch, row["path"], use_cache)
        if df is None:
            continue
        key = f"{row['season']}/{row['table']}"
        print(f"\n  {key}   ({row['path']})")
        print(f"    shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

        date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in df.columns), None)
        if date_col:
            d = pd.to_datetime(df[date_col], errors="coerce", format="mixed", utc=True)
            if d.notna().any():
                print(f"    dates: {d.min().date()} -> {d.max().date()}  (via `{date_col}`)")

        print(f"    {'column':<34}{'dtype':<12}{'non-null':>10}{'unique':>9}   example")
        print("    " + "-" * 88)
        for c in df.columns:
            s = df[c]
            nn = f"{100 * s.notna().mean():.0f}%"
            ex = s.dropna()
            ex = str(ex.iloc[0])[:26] if len(ex) else ""
            print(f"    {c[:33]:<34}{str(s.dtype):<12}{nn:>10}{s.nunique():>9}   {ex}")

        out[key] = {
            "path": row["path"],
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
        }
    return out


def profile_deep(branch: str, inv: pd.DataFrame, use_cache: bool, season: str | None) -> dict:
    """Concatenate every gameweek file of the key tables for true coverage numbers."""
    rule("DEEP PROFILE  (all gameweeks concatenated)")
    out = {}
    target = inv if season is None else inv[inv["season"] == season]

    for (seas, table), grp in target.groupby(["season", "table"]):
        if table not in KEY_TABLES:
            continue
        gw_files = grp[grp["scope"] == "gameweek"]
        if gw_files.empty:
            continue
        print(f"\n  {seas}/{table}: fetching {len(gw_files)} gameweek files ...")
        frames = []
        for _, row in gw_files.sort_values("gameweek").iterrows():
            df = fetch_csv(branch, row["path"], use_cache)
            if df is not None and len(df):
                df["_gameweek"] = row["gameweek"]
                df["_competition"] = row["competition"]
                frames.append(df)
        if not frames:
            continue
        full = pd.concat(frames, ignore_index=True)
        print(f"    total: {len(full):,} rows across {full['_gameweek'].nunique()} gameweeks")

        date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in full.columns), None)
        if date_col:
            d = pd.to_datetime(full[date_col], errors="coerce", format="mixed", utc=True)
            if d.notna().any():
                span = (d.max() - d.min()).days
                print(f"    dates: {d.min().date()} -> {d.max().date()}  ({span} days)")
        for c in ("match_id", "player_id", "team_id", "id"):
            if c in full.columns:
                print(f"    distinct {c}: {full[c].nunique():,}")
        if "_competition" in full.columns and full["_competition"].nunique() > 1:
            print(f"    by competition: {full['_competition'].value_counts().to_dict()}")

        out[f"{seas}/{table}"] = {
            "rows": int(len(full)),
            "gameweeks": int(full["_gameweek"].nunique()),
            "columns": list(full.columns),
        }
    return out


# --- main ---

def main() -> None:
    ap = argparse.ArgumentParser(description="Inventory and profile the FPL-Core-Insights dataset.")
    ap.add_argument("--season", help="restrict profiling to one season, e.g. 2025-2026")
    ap.add_argument("--deep", action="store_true",
                    help="also concatenate every gameweek of the key tables")
    ap.add_argument("--no-cache", action="store_true", help="ignore the local .cache/ directory")
    args = ap.parse_args()

    use_cache = not args.no_cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    rule("UPSTREAM REPOSITORY", "#")
    meta = get_repo_meta()
    for k, v in meta.items():
        print(f"  {k:<16}: {v}")
    print(f"\n  addressed by numeric id {REPO_ID} so a future rename cannot break this")
    print("  (v1 hardcoded 'olbauday/FPL-Elo-Insights', which is now a redirect)")

    print("\n  fetching full repo tree (1 API call) ...")
    tree = get_tree(meta["default_branch"])
    inv = build_inventory(tree)
    print(f"  {len(tree):,} files in repo, {len(inv):,} of them CSVs under data/")

    report_seasons(inv)
    report_layout(inv)
    report_table_matrix(inv)
    report_competitions(inv)

    schema = profile_schema(meta["default_branch"], inv, use_cache, args.season)
    deep = profile_deep(meta["default_branch"], inv, use_cache, args.season) if args.deep else {}

    inv_path = REPORT_DIR / "inventory.csv"
    inv.sort_values(["season", "competition", "table", "gameweek"]).to_csv(inv_path, index=False)

    report_path = REPORT_DIR / "dataset_report.json"
    report_path.write_text(json.dumps(
        {"repo": meta, "n_csv_files": len(inv), "schema": schema, "deep": deep},
        indent=2), encoding="utf-8")

    rule("SAVED")
    print(f"  {inv_path}   (every CSV, one row each)")
    print(f"  {report_path}   (schemas + profile)")
    print(f"  {CACHE_DIR}/   (downloaded files, reused on next run)")
    print(f"\n  done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
