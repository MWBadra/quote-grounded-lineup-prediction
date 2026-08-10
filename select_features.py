"""Applies the manual feature selection, then verifies that no redundancy survives
and re-measures the model on the pruned set.

Rationale is recorded per feature so the methods section can be written off this
file instead of from memory.

python select_features.py
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

# --- the kept feature set ---
KEEP = {
    # selection history. one recency measure only, started_last_1/3/5 are all
    # >=0.94 correlated with start_rate_last_5 and add nothing.
    "start_rate_last_5":               "recency of selection, the single strongest signal",
    "consecutive_starts":              "run length, distinct from rate",

    # NOTE: f5_minutes_played was also dropped. It sat at r = 0.940 with
    # start_rate_last_5, above the 0.90 redundancy bar, and a 5-seed test showed
    # removing it costs nothing (accuracy +0.0018, ROC-AUC -0.0021, both inside the
    # +/-0.0017 seed noise). start_rate_last_5 is kept instead: higher correlation
    # with the target (0.687 vs 0.589) and no missing values against its 25%.

    # availability. KEPT AGAINST THE MARGINAL-CORRELATION RULE: r is only +0.116
    # overall because just 5.2% of rows are doubtful, but within that group the
    # starter rate falls from 0.410 to 0.191.
    "availability_score":              "graded injury/doubt, 25/50/75/100",

    # position
    "player_position":                 "GK / D-M-F x CENTRE-WIDE, from prior coordinates",

    # defending. keep the composite, drop its parts
    "clean_sheets":                    "team defensive record while on pitch",
    "recoveries":                      "ball recoveries, all positions",
    "clearances_blocks_interceptions": "composite defensive actions, all positions",

    # passing. the rate not the count, since the count just tracks minutes
    "f5_accurate_passes_percent":      "passing accuracy, minutes-independent",

    # attacking
    "goals_scored":                    "season goals",
    "assists":                         "season assists",
    "f5_chances_created":              "recent creation",
    "f5_total_shots":                  "recent shot volume",

    # physical / duel
    "f5_fouls_committed":              "engagement proxy",

    # discipline
    "yellow_cards":                    "season yellows",
    "red_cards":                       "suspension mechanism; confounding controlled by the recency features",

    # squad status. is_new_to_league is preferred over is_new_signing: a player
    # moving within the league already has usable history, so arriving from
    # outside it is the condition that actually blinds the tabular features.
    # The two differ on only 4 rows this season (Sancho, Garnacho, Marc Guiu,
    # Bobb) but will diverge properly once a second season is added.
    "is_new_to_league":                "no prior Premier League appearance",

    # goalkeeper. KEPT AGAINST THE MARGINAL-CORRELATION RULE: structurally zero
    # for outfielders so the global r understates it badly. within goalkeepers
    # gk_saves is +0.640. gk_clean_sheets_per_90 was dropped because plain
    # clean_sheets scores +0.648 on the same rows, i.e. strictly better.
    "gk_saves":                        "GK shot-stopping volume",
}

DROPPED = {
    "started_last_1/3/5":        "r >= 0.94 with start_rate_last_5",
    "minutes, minutes_season":   "exact duplicates (r = 1.000); minutes tracked by f5_minutes_played",
    "starts, starts_season":     "restate start_rate_last_5",
    "form, points_per_game":     "FPL scoring artefacts, not football quantities",
    "influence, bps, total_points, ict_index, creativity, threat, bonus":
                                 "FPL scoring composites, all inside the minutes cluster",
    "selected_by_percent, now_cost": "FPL game-economy fields, not team-selection inputs",
    "defensive_contribution":    "count of defensive actions; scales with minutes, |r|>=0.9 with it",
    "goals_conceded":            "inverse of clean_sheets",
    "tackles, f5_tackles_won, f5_clearances, f5_interceptions, f5_blocks":
                                 "components of clearances_blocks_interceptions",
    "f5_accurate_passes, f5_final_third_passes": "counts that track minutes; the percentage is kept",
    "all expected_* / f5_xg / f5_xa / f5_xgot_faced / f5_goals_prevented":
                                 "modelled rather than observed quantities",
    "was_unused_sub_last_match": "restates start_rate_last_5",
    "yellows_from_ban":          "derived from yellow_cards",
    "has_history, is_new_signing": "r >= 0.999 with is_new_to_league",
    "f5_touches_opposition_box": "0.772 with f5_total_shots, same attacking-presence signal",
    "f5_aerial_duels_won":       "marginal contribution below the retention bar",
    "gk_clean_sheets_per_90":    "clean_sheets scores higher on the same GK rows (+0.648 vs +0.543)",
    "position_is_inferred":      "bookkeeping flag, restates has_history",
    "is_set_piece_taker, *_order": "duty flags, largely a fame proxy",
    "days_since_last_start":     "restates recency",
    "days_since_prev_match, days_until_next_match": "no signal (|r| < 0.01)",
    "opp_elo, opp_strength_*, is_home": "no marginal signal (|r| < 0.01)",
    "f5_duels_won, f5_dribbled_past, f5_successful_dribbles, f5_was_fouled, f5_shots_on_target":
                                 "below the shots_on_target cutoff or redundant",
    "f5_recoveries, f5_saves, f5_goals_conceded": "restate their season-level counterparts",
    "gk_saves_per_90, gk_goals_conceded_per_90, gk_penalties_saved":
                                 "redundant with the two kept GK features",
}


def rule(t, ch="="):
    print("\n" + ch * 80)
    print(t)
    print(ch * 80)


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    y = df["is_starter"].astype(int)
    keep = [c for c in KEEP if c in df.columns]
    missing = [c for c in KEEP if c not in df.columns]

    rule("SELECTED FEATURE SET", "#")
    print(f"  kept {len(keep)} of 84 features")
    if missing:
        print(f"  !! not found in dataset: {missing}")

    corr = {}
    for c in keep:
        if pd.api.types.is_numeric_dtype(df[c]):
            m = df[c].notna()
            corr[c] = stats.pointbiserialr(y[m], df[c][m])[0]
    print(f"\n  {'feature':<32}{'r':>8}   rationale")
    print("  " + "-" * 92)
    for c in keep:
        r = f"{corr[c]:+.3f}" if c in corr else "  cat"
        print(f"  {c:<32}{r:>8}   {KEEP[c]}")

    rule("DROPPED — 64 features")
    for k, v in DROPPED.items():
        print(f"  {k}\n      {v}")

    # --- redundancy audit on the survivors ---
    num = [c for c in keep if pd.api.types.is_numeric_dtype(df[c])]
    C = df[num].corr()
    pairs = (C.abs().stack().reset_index()
             .rename(columns={"level_0": "a", "level_1": "b", 0: "r"}))
    pairs = pairs[(pairs.a < pairs.b) & (pairs.r >= 0.70)].sort_values("r", ascending=False)
    rule("REDUNDANCY AUDIT ON SURVIVORS  (|r| >= 0.70)")
    if pairs.empty:
        print("  clean — no surviving pair reaches 0.70")
    else:
        for _, x in pairs.iterrows():
            flag = "  <-- STILL REDUNDANT" if x.r >= 0.90 else ""
            print(f"  {x.a:<32}{x.b:<32}{x.r:>7.3f}{flag}")
    print(f"\n  max |r| among survivors: {pairs.r.max() if not pairs.empty else 0:.3f}")

    # --- model on the pruned set ---
    rule("MODEL ON THE PRUNED SET  (chronological 70/30)")
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from xgboost import XGBClassifier

    X = df[keep].copy()
    for c in keep:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    tr, va = df.gameweek < cut, df.gameweek >= cut

    def run(cols, label):
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=42)
        m.fit(X.loc[tr, cols], y[tr])
        p = m.predict_proba(X.loc[va, cols])[:, 1]
        a, u, f = (accuracy_score(y[va], p >= .5), roc_auc_score(y[va], p),
                   f1_score(y[va], p >= .5))
        print(f"  {label:<34}{a:>9.4f}{u:>9.4f}{f:>9.4f}")
        return m, a, u, f

    print(f"  split at GW{cut}: train {tr.sum():,} / val {va.sum():,}")
    print(f"\n  {'model':<34}{'acc':>9}{'ROC-AUC':>9}{'F1':>9}")
    print("  " + "-" * 62)
    m, *_ = run(keep, f"pruned ({len(keep)} features)")
    run(["start_rate_last_5"], "start_rate_last_5 alone")

    imp = pd.Series(m.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    imp = 100 * imp / imp.sum()
    print(f"\n  {'feature':<34}{'gain %':>9}")
    print("  " + "-" * 45)
    for k, v in imp.items():
        print(f"  {k:<34}{v:>9.2f}")
    print(f"\n  top feature carries {imp.iloc[0]:.1f}% of gain (was 67.0% before pruning)")

    out = df[["season", "gameweek", "match_id", "team_code", "team_name", "player_id",
              "web_name", "is_starter", "formation", "in_matchday_squad"] + keep]
    p = HERE / "dataset" / "pl_2025_2026_selected.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    rule("SAVED")
    print(f"  {p}   ({out.shape[0]:,} x {out.shape[1]})")


if __name__ == "__main__":
    main()
