"""Export the correlation payload the HTML report reads."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

METADATA = ["season", "gameweek", "match_id", "kickoff_time", "team_code", "team_name",
            "opponent_code", "opponent_name", "player_id", "player_code", "web_name",
            "fpl_position", "status"]
LABELS = ["is_starter", "formation", "in_matchday_squad", "minutes_played"]
BLOCKS = {
    "position": ["player_position", "position_is_inferred"],
    "availability": ["is_available", "availability_score"],
    "history": ["started_last_1", "started_last_3", "started_last_5", "start_rate_last_5",
                "starts_season", "minutes_season", "consecutive_starts",
                "days_since_last_start", "was_unused_sub_last_match", "has_history"],
    "rest": ["days_since_prev_match", "days_until_next_match"],
    "transfer": ["is_new_signing", "is_new_to_league"],
    "opponent": ["opp_elo", "opp_strength_overall", "opp_strength_attack",
                 "opp_strength_defence"],
    "context": ["is_home"],
}


def block_of(c):
    for k, v in BLOCKS.items():
        if c in v:
            return k
    if c.startswith("f5_"):
        return "form_last5"
    if c.startswith("gk_"):
        return "goalkeeper"
    return "season_stats"


df = pd.read_csv(HERE / "dataset" / "pl_2025_2026_starting_xi.csv", encoding="utf-8-sig")
y = df["is_starter"].astype(int)
feats = [c for c in df.columns if c not in METADATA + LABELS
         and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]

# correlation with the target
tgt = []
for c in feats:
    m = df[c].notna()
    r, p = stats.pointbiserialr(y[m], df[c][m])
    tgt.append({"f": c, "b": block_of(c), "r": round(float(r), 4),
                "p": float(p), "null": round(100 * float(df[c].isna().mean()), 1)})
tgt.sort(key=lambda d: -abs(d["r"]))

# feature x feature. cluster the ordering, otherwise the redundant groups are
# scattered all over the matrix and you can't see them
C = df[feats].corr().fillna(0.0)
D = np.clip(1.0 - C.abs().to_numpy(), 0, None)
np.fill_diagonal(D, 0.0)
order = leaves_list(linkage(squareform(D, checks=False), method="average"))
cols = [feats[i] for i in order]
M = C.loc[cols, cols].to_numpy()

# redundancy pairs
pairs = []
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if abs(M[i, j]) >= 0.90:
            pairs.append({"a": cols[i], "b": cols[j], "r": round(float(M[i, j]), 3)})
pairs.sort(key=lambda d: -abs(d["r"]))

payload = {
    "n_rows": int(len(df)),
    "n_team_matches": int(df.groupby(["match_id", "team_code"]).ngroups),
    "base_rate": round(float(y.mean()), 4),
    "features": cols,
    "blocks": [block_of(c) for c in cols],
    # ints scaled by 100, keeps the json down to a sane size
    "matrix": [[int(round(v * 100)) for v in row] for row in M],
    "target": tgt,
    "pairs": pairs,
}
out = HERE / "reports" / "corr_payload.json"
out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print(f"features {len(cols)} | pairs>=0.90 {len(pairs)} | {out.stat().st_size/1024:.0f} KB")
print("blocks:", {b: payload["blocks"].count(b) for b in set(payload["blocks"])})
