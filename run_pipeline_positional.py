"""The positional architecture: instead of one model scoring every player on the
same features, each position group is scored by a model trained only on players
of that group, using the statistics that actually describe the job.

  GK   shot-stopping and clean sheets
  DEF  clearances/blocks/interceptions, clean sheets, recoveries, discipline
  MID  passing accuracy, chances created, plus both defensive and attacking work
  FWD  goals, assists, shots, chances created

All four groups also get the shared availability and selection-history columns,
since those drive selection whatever position you play, plus the LLM signals.

Stage 4 assigns the eleven the same way as before. Both formation conditions get
run, the forecast one and the true shape handed over.

python run_pipeline_positional.py
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from run_pipeline_v2 import (ELIGIBILITY, LLM, SEEDS, TABULAR, evaluate_squads,
                             forecast_formations, learn_templates, rule,
                             solve_lineup)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"

# Governs selection whatever the position: is he fit, is he in the side lately,
# and what does the pre-match reporting say.
SHARED = ["start_rate_last_5", "consecutive_starts", "availability_score",
          "is_new_to_league", "player_position"] + LLM

# Position-specific evidence. A centre-back and a striker are not competing on
# the same qualities, and a single global model has to spend capacity learning
# to ignore each one's irrelevant columns.
POSITION_FEATURES = {
    "GK":  ["gk_saves", "clean_sheets", "recoveries"],
    "DEF": ["clearances_blocks_interceptions", "clean_sheets", "recoveries",
            "yellow_cards", "red_cards", "f5_accurate_passes_percent",
            "f5_fouls_committed"],
    "MID": ["f5_accurate_passes_percent", "f5_chances_created", "recoveries",
            "clearances_blocks_interceptions", "assists", "goals_scored",
            "f5_total_shots", "yellow_cards", "f5_fouls_committed"],
    "FWD": ["goals_scored", "assists", "f5_total_shots", "f5_chances_created",
            "f5_accurate_passes_percent", "yellow_cards"],
}

GROUP_OF = {"GK": "GK", "D-CENTRE": "DEF", "D-WIDE": "DEF",
            "M-CENTRE": "MID", "M-WIDE": "MID",
            "F-CENTRE": "FWD", "F-WIDE": "FWD"}


def encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X


def fit_global(df: pd.DataFrame, cols: list[str], cut: int) -> np.ndarray:
    """One model over every player -- the v2 baseline architecture."""
    X, y = encode(df, cols), df.is_starter.astype(int)
    tr, va = df.gameweek < cut, df.gameweek >= cut
    preds = []
    for sd in SEEDS:
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=sd).fit(X[tr], y[tr])
        preds.append(m.predict_proba(X[va])[:, 1])
    return np.mean(preds, axis=0)


def fit_positional(df: pd.DataFrame, cut: int, use_llm: bool = True):
    """
    One model per position group, each on its own rows and its own features.

    Returns validation probabilities aligned to df[val] order, plus per-group
    metrics so it is visible which positions the split actually helps.
    """
    shared = SHARED if use_llm else [c for c in SHARED if c not in LLM]
    va_mask = df.gameweek >= cut
    out = pd.Series(np.nan, index=df.index[va_mask])
    per_group = {}

    for grp in ["GK", "DEF", "MID", "FWD"]:
        rows = df[df.player_position.map(GROUP_OF) == grp]
        if rows.empty:
            continue
        cols = shared + POSITION_FEATURES[grp]
        cols = list(dict.fromkeys(cols))            # de-duplicate, keep order
        X, y = encode(rows, cols), rows.is_starter.astype(int)
        tr, va = rows.gameweek < cut, rows.gameweek >= cut
        if tr.sum() < 50 or va.sum() == 0:
            continue
        preds, met = [], []
        for sd in SEEDS:
            m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=sd).fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            preds.append(p)
            met.append((accuracy_score(y[va], p >= .5),
                        roc_auc_score(y[va], p) if y[va].nunique() > 1 else np.nan,
                        f1_score(y[va], p >= .5)))
        p_mean = np.mean(preds, axis=0)
        out.loc[rows.index[va]] = p_mean
        a = np.array(met)
        per_group[grp] = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
                          "acc": a[:, 0].mean(), "auc": np.nanmean(a[:, 1]),
                          "f1": a[:, 2].mean(), "n_feat": len(cols)}
    return out.to_numpy(), per_group


def row_metrics(y, p):
    return (accuracy_score(y, p >= .5), roc_auc_score(y, p), f1_score(y, p >= .5))


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()
    yv = val.is_starter.astype(int)
    templates = learn_templates(df)
    fc = forecast_formations(df)

    rule("POSITIONAL ARCHITECTURE", "#")
    print(f"  split GW{cut} | val rows {len(val):,} | "
          f"team-matches {val.groupby(['match_id','team_code']).ngroups}")
    print(f"  position groups: " +
          ", ".join(f"{g}={int((df.player_position.map(GROUP_OF)==g).sum()):,}"
                    for g in ["GK", "DEF", "MID", "FWD"]))

    # --- per-group models ---
    p_pos, groups = fit_positional(df, cut, use_llm=True)
    p_pos_nollm, groups_nollm = fit_positional(df, cut, use_llm=False)
    p_glob = fit_global(df, TABULAR + LLM, cut)
    p_glob_nollm = fit_global(df, TABULAR, cut)

    rule("PER-POSITION MODELS  (each trained only on its own players)")
    print(f"  {'group':<7}{'feats':>7}{'train':>9}{'val':>8}"
          f"{'accuracy':>11}{'ROC-AUC':>10}{'F1':>9}")
    print("  " + "-" * 62)
    for g, m in groups.items():
        print(f"  {g:<7}{m['n_feat']:>7}{m['n_train']:>9,}{m['n_val']:>8,}"
              f"{m['acc']:>11.4f}{m['auc']:>10.4f}{m['f1']:>9.4f}")

    rule("ROW-LEVEL: POSITIONAL vs GLOBAL")
    print(f"  {'architecture':<34}{'accuracy':>11}{'ROC-AUC':>11}{'F1':>10}")
    print("  " + "-" * 66)
    for label, p in [("global, tabular only", p_glob_nollm),
                     ("global, + LLM", p_glob),
                     ("positional, tabular only", p_pos_nollm),
                     ("positional, + LLM", p_pos)]:
        a, u, f = row_metrics(yv, p)
        print(f"  {label:<34}{a:>11.4f}{u:>11.4f}{f:>10.4f}")

    # --- squad level, both formation conditions ---
    rule("SQUAD LEVEL  (11 from ~27)")
    print(f"  {'architecture':<26}{'formation':<12}{'n':>5}"
          f"{'constrained':>13}{'>=9/11':>9}{'11/11':>8}")
    print("  " + "-" * 74)
    best = {}
    for label, p in [("global, tabular only", p_glob_nollm),
                     ("global, + LLM", p_glob),
                     ("positional, tabular only", p_pos_nollm),
                     ("positional, + LLM", p_pos)]:
        val["p"] = p
        for mode in ("markov", "oracle"):
            r = evaluate_squads(val, templates, fc, mode)
            best[(label, mode)] = r
            print(f"  {label:<26}{mode:<12}{r['n']:>5}{100*r['constrained']:>12.2f}%"
                  f"{100*r['ge9']:>8.1f}%{100*r['perfect']:>7.1f}%")

    # --- who wins, and where the llm helps ---
    rule("SUMMARY")
    g_m = best[("global, + LLM", "markov")]
    p_m = best[("positional, + LLM", "markov")]
    g_o = best[("global, + LLM", "oracle")]
    p_o = best[("positional, + LLM", "oracle")]
    print(f"  positional vs global (forecast formation, +LLM):")
    print(f"      constrained {100*g_m['constrained']:.2f}% -> {100*p_m['constrained']:.2f}%"
          f"   ({100*(p_m['constrained']-g_m['constrained']):+.2f}pp)")
    print(f"      >=9/11      {100*g_m['ge9']:.1f}% -> {100*p_m['ge9']:.1f}%"
          f"   ({100*(p_m['ge9']-g_m['ge9']):+.1f}pp)")
    print(f"\n  cost of forecasting the formation instead of being given it:")
    for lab, m, o in [("global", g_m, g_o), ("positional", p_m, p_o)]:
        print(f"      {lab:<12} {100*m['constrained']:.2f}% vs {100*o['constrained']:.2f}%"
              f"   ({100*(m['constrained']-o['constrained']):+.2f}pp)")
    print(f"\n  LLM contribution within each architecture (forecast formation):")
    for lab in ["global", "positional"]:
        a = best[(f"{lab}, tabular only", "markov")]
        b = best[(f"{lab}, + LLM", "markov")]
        print(f"      {lab:<12} constrained {100*(b['constrained']-a['constrained']):+.2f}pp"
              f"   >=9/11 {100*(b['ge9']-a['ge9']):+.1f}pp"
              f"   11/11 {100*(b['perfect']-a['perfect']):+.1f}pp")
    print(f"\n  best X/11 distribution: "
          f"{dict(sorted(p_m['dist'].items(), reverse=True))}")


if __name__ == "__main__":
    main()
