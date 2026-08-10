"""Feature report for the v2 tabular dataset.

Mostly here to catch redundancy. v1's headline ablation was distorted because
games_started_season and games_played_season were near-copies of
minutes_played_season (r = 0.993 and 0.922), which left the tabular baseline
and the hybrid model not really comparable. Catching those clusters before we
model anything is the point.

python analyze_features.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_starting_xi.csv"
OUT = HERE / "reports"

TARGET = "is_starter"

# Columns that must never be treated as features.
METADATA = ["season", "gameweek", "match_id", "kickoff_time", "team_code",
            "team_name", "opponent_code", "opponent_name", "player_id",
            "player_code", "web_name", "fpl_position", "status"]
LABELS = ["is_starter", "formation", "in_matchday_squad", "minutes_played"]

BLOCKS = {
    "position":     ["player_position", "position_is_inferred"],
    "availability": ["is_available", "availability_score"],
    "history":      ["started_last_1", "started_last_3", "started_last_5",
                     "start_rate_last_5", "starts_season", "minutes_season",
                     "consecutive_starts", "days_since_last_start",
                     "was_unused_sub_last_match", "has_history"],
    "rest":         ["days_since_prev_match", "days_until_next_match"],
    "transfer":     ["is_new_signing", "is_new_to_league"],
    "opponent":     ["opp_elo", "opp_strength_overall", "opp_strength_attack",
                     "opp_strength_defence"],
    "context":      ["is_home"],
}


def block_of(col: str) -> str:
    for name, cols in BLOCKS.items():
        if col in cols:
            return name
    if col.startswith("f5_"):
        return "form_last5"
    if col.startswith("gk_"):
        return "goalkeeper"
    return "season_stats"


def rule(title: str, ch: str = "=") -> None:
    print("\n" + ch * 84)
    print(title)
    print(ch * 84)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df[TARGET].astype(int)

    feats = [c for c in df.columns if c not in METADATA + LABELS]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]

    rule("DATASET", "#")
    print(f"  rows {len(df):,} | features {len(feats)} ({len(num)} numeric, {len(cat)} categorical)")
    print(f"  target `{TARGET}`: {y.mean():.4f} positive ({y.sum():,} of {len(y):,})")
    print(f"  team-matches {df.groupby(['match_id','team_code']).ngroups}")

    # --- 1. inventory ---
    inv = pd.DataFrame({
        "feature": num,
        "block": [block_of(c) for c in num],
        "null_pct": [round(100 * df[c].isna().mean(), 1) for c in num],
        "n_unique": [df[c].nunique() for c in num],
        "mean": [round(float(df[c].mean()), 3) if df[c].notna().any() else np.nan for c in num],
        "std": [round(float(df[c].std()), 3) if df[c].notna().any() else np.nan for c in num],
    })

    dead = inv[(inv.n_unique <= 1) | (inv.null_pct == 100)]
    rule("DEAD FEATURES (constant or fully null) — drop these")
    print("  none" if dead.empty else dead[["feature", "block", "null_pct", "n_unique"]].to_string(index=False))
    drop = set(dead.feature)
    live = [c for c in num if c not in drop]

    # --- 2. correlation with target ---
    rows = []
    for c in live:
        s = df[c]
        m = s.notna()
        if m.sum() < 50 or s[m].nunique() < 2:
            continue
        r, p = stats.pointbiserialr(y[m], s[m])
        rows.append((c, block_of(c), r, p, round(100 * s.isna().mean(), 1)))
    corr = pd.DataFrame(rows, columns=["feature", "block", "r", "p", "null_pct"])
    corr["abs_r"] = corr.r.abs()
    corr = corr.sort_values("abs_r", ascending=False)

    rule(f"CORRELATION WITH `{TARGET}`  — top 30 by |r|")
    print(f"  {'feature':<34}{'block':<14}{'r':>8}{'p':>11}{'null%':>8}")
    print("  " + "-" * 76)
    for _, x in corr.head(30).iterrows():
        print(f"  {x.feature:<34}{x.block:<14}{x.r:>+8.3f}{x.p:>11.2e}{x.null_pct:>8.1f}")

    rule("WEAKEST 15 — candidates to drop")
    for _, x in corr.tail(15).iterrows():
        print(f"  {x.feature:<34}{x.block:<14}{x.r:>+8.3f}{x.p:>11.2e}")

    rule("SIGNAL BY BLOCK")
    b = corr.groupby("block").agg(n=("feature", "size"), mean_abs_r=("abs_r", "mean"),
                                  max_abs_r=("abs_r", "max")).round(3)
    print(b.sort_values("max_abs_r", ascending=False).to_string())

    # --- 3. redundancy ---
    C = df[live].corr()
    # Take the strict upper triangle rather than mutating the diagonal: under
    # copy-on-write pandas hands back a read-only .values buffer.
    pairs = (C.abs().stack().reset_index()
             .rename(columns={"level_0": "a", "level_1": "b", 0: "abs_r"}))
    pairs = pairs[pairs.a < pairs.b].sort_values("abs_r", ascending=False)

    rule("REDUNDANT PAIRS  |r| >= 0.90  — this is what broke v1's ablation")
    hi = pairs[pairs.abs_r >= 0.90]
    if hi.empty:
        print("  none")
    else:
        tmap = dict(zip(corr.feature, corr.r))
        print(f"  {'feature A':<32}{'feature B':<32}{'|r|':>7}{'rA':>7}{'rB':>7}")
        print("  " + "-" * 84)
        for _, x in hi.head(40).iterrows():
            print(f"  {x.a:<32}{x.b:<32}{x.abs_r:>7.3f}"
                  f"{tmap.get(x.a, np.nan):>+7.3f}{tmap.get(x.b, np.nan):>+7.3f}")
        print(f"\n  {len(hi)} pairs at |r| >= 0.90")

    # greedy clustering: keep the member most correlated with the target
    rule("REDUNDANCY CLUSTERS  |r| >= 0.90  — keep one per cluster")
    adj = {c: set() for c in live}
    for _, x in hi.iterrows():
        adj[x.a].add(x.b)
        adj[x.b].add(x.a)
    seen, clusters = set(), []
    for c in live:
        if c in seen or not adj[c]:
            continue
        stack, comp = [c], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v] - seen)
        clusters.append(comp)
    tmap = dict(zip(corr.feature, corr.abs_r))
    keep_drop = []
    for comp in clusters:
        ranked = sorted(comp, key=lambda c: -tmap.get(c, 0))
        print(f"\n  KEEP  {ranked[0]:<34} (|r| with target {tmap.get(ranked[0], 0):.3f})")
        for d in ranked[1:]:
            print(f"    drop {d:<34} (|r| {tmap.get(d, 0):.3f})")
            keep_drop.append({"drop": d, "in_favour_of": ranked[0]})
    if not clusters:
        print("  no clusters")

    # --- 4. categorical ---
    rule("CATEGORICAL FEATURES")
    for c in cat:
        t = df.groupby(c)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        ct = pd.crosstab(df[c], y)
        chi2, p, _, _ = stats.chi2_contingency(ct)
        v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
        print(f"\n  {c}   (Cramer's V = {v:.3f}, p = {p:.2e})")
        print(t.assign(mean=t["mean"].round(3)).to_string())

    # --- 5. model-based importance ---
    rule("XGBOOST GAIN IMPORTANCE  (chronological 70/30 split)")
    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        X = df[live].copy()
        for c in cat:
            X[c] = df[c].astype("category").cat.codes
        gws = sorted(df.gameweek.unique())
        cut = gws[int(len(gws) * 0.7)]
        tr, va = df.gameweek < cut, df.gameweek >= cut
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=42)
        m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[va])[:, 1]
        print(f"  split at GW{cut}: train {tr.sum():,} / val {va.sum():,}")
        print(f"  accuracy {accuracy_score(y[va], pr >= .5):.4f} | "
              f"ROC-AUC {roc_auc_score(y[va], pr):.4f} | F1 {f1_score(y[va], pr >= .5):.4f}")
        imp = (pd.Series(m.get_booster().get_score(importance_type="gain"))
               .sort_values(ascending=False))
        imp = 100 * imp / imp.sum()
        print(f"\n  {'feature':<34}{'gain %':>9}   block")
        print("  " + "-" * 66)
        for k, v in imp.head(25).items():
            print(f"  {k:<34}{v:>9.2f}   {block_of(k)}")
        print(f"\n  top 5 features carry {imp.head(5).sum():.1f}% of total gain")
        print(f"  top 15 carry {imp.head(15).sum():.1f}%")
        imp.to_csv(OUT / "feature_importance.csv", header=["gain_pct"])
    except Exception as e:  # noqa: BLE001
        print(f"  skipped: {e}")

    # --- 6. save ---
    corr.round(4).to_csv(OUT / "feature_correlations.csv", index=False)
    C.round(4).to_csv(OUT / "correlation_matrix.csv")
    pd.DataFrame(keep_drop).to_csv(OUT / "redundancy_drop_list.csv", index=False)
    inv.to_csv(OUT / "feature_inventory.csv", index=False)

    rule("SAVED")
    for f in ["feature_correlations.csv", "correlation_matrix.csv",
              "redundancy_drop_list.csv", "feature_inventory.csv",
              "feature_importance.csv"]:
        print(f"  reports/{f}")


if __name__ == "__main__":
    main()
