"""The positional idea again, but without paying for it in sample size.

Hard-splitting by position didn't work, each model only saw 13-37% of the data
(see run_pipeline_positional). This gets the same compare-defenders-to-defenders
logic out of one model on all 13,746 rows: swap each statistic for the player's
percentile within his position group, in his own squad, for that fixture.

Two things you get for free. The column is position-relative by construction, so
pct_clearances ranks a defender against defenders and a midfielder against
midfielders without splitting anything. And it encodes competition for the
shirt, which a raw value simply cannot: the manager picks the best available
centre-back, not every centre-back over some threshold, and "second best
defender in today's squad" is the quantity that decides it.

No leak. All source columns are pre-match already and the ranking only ever
looks inside the squad available for that fixture.

python positional_percentiles.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from run_pipeline_positional import GROUP_OF
from run_pipeline_v2 import (LLM, TABULAR, evaluate_squads, fit_predict,
                             forecast_formations, learn_templates, rule)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"

# Ranked within (fixture, squad, position group). Spans all four position
# families on purpose. A midfielder's clearance percentile among midfielders
# says just as much as a defender's does among defenders.
PCT_STATS = [
    "start_rate_last_5", "consecutive_starts",              # selection history
    "clean_sheets", "recoveries", "clearances_blocks_interceptions",
    "f5_accurate_passes_percent", "f5_chances_created",
    "goals_scored", "assists", "f5_total_shots",
    "gk_saves",
]


def add_percentiles(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["_grp"] = df.player_position.map(GROUP_OF)
    key = ["match_id", "team_code", "_grp"]
    made = []
    g = df.groupby(key, sort=False)
    for c in PCT_STATS:
        if c not in df.columns:
            continue
        name = f"pct_{c}"
        # rank in (0, 1]. a group of one collapses to 1.0, which is right: if
        # he's the only option at his position then he is the best one.
        df[name] = g[c].rank(pct=True, method="average")
        made.append(name)
    # how many rivals for the shirt. sets the scale a percentile is read on
    df["pos_group_size"] = g["player_id"].transform("size")
    made.append("pos_group_size")
    return df.drop(columns="_grp"), made


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df, pct_cols = add_percentiles(df)
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()
    templates = learn_templates(df)
    fc = forecast_formations(df)

    rule("POSITION-RELATIVE PERCENTILE FEATURES", "#")
    print(f"  built {len(pct_cols)} features, ranked within (fixture, squad, position group)")
    print(f"  split GW{cut} | val {len(val):,} rows | "
          f"{val.groupby(['match_id','team_code']).ngroups} team-matches")
    sizes = df.groupby(["match_id", "team_code", df.player_position.map(GROUP_OF)]).size()
    print(f"  players per position group: mean {sizes.mean():.1f} "
          f"(min {sizes.min()}, max {sizes.max()})")

    # sanity: does rank-within-squad separate starters better than the raw value?
    rule("DO THE PERCENTILES CARRY SIGNAL THE RAW VALUES DO NOT?")
    print(f"  {'statistic':<36}{'raw |r|':>10}{'pct |r|':>10}{'change':>10}")
    print("  " + "-" * 68)
    y = df.is_starter.astype(int)
    for c in PCT_STATS:
        if c not in df.columns or f"pct_{c}" not in df.columns:
            continue
        a = abs(df[c].corr(y))
        b = abs(df[f"pct_{c}"].corr(y))
        print(f"  {c:<36}{a:>10.3f}{b:>10.3f}{b-a:>+10.3f}")

    # --- configurations ---
    configs = {
        "global tabular (17)":              TABULAR,
        "global tabular + LLM":             TABULAR + LLM,
        "+ percentiles (no LLM)":           TABULAR + pct_cols,
        "+ percentiles + LLM":              TABULAR + pct_cols + LLM,
        "percentiles only + shared":        (pct_cols + ["player_position",
                                             "availability_score", "is_new_to_league"]),
    }
    rule("ROW-LEVEL  (5 seeds)")
    print(f"  {'configuration':<32}{'accuracy':>18}{'ROC-AUC':>18}{'F1':>9}")
    print("  " + "-" * 78)
    probs, mets = {}, {}
    for name, cols in configs.items():
        cols = list(dict.fromkeys(cols))
        p, m = fit_predict(df, cols, cut)
        probs[name], mets[name] = p, m
        print(f"  {name:<32}{m['acc']:>10.4f}+/-{m['acc_sd']:.4f}"
              f"{m['auc']:>10.4f}+/-{m['auc_sd']:.4f}{m['f1']:>9.4f}")

    b = mets["global tabular + LLM"]
    n = mets["+ percentiles + LLM"]
    rule("GAIN FROM PERCENTILES")
    print(f"  vs global tabular + LLM:  acc {n['acc']-b['acc']:+.4f}"
          f"   auc {n['auc']-b['auc']:+.4f}   f1 {n['f1']-b['f1']:+.4f}")
    b0 = mets["global tabular (17)"]
    n0 = mets["+ percentiles (no LLM)"]
    print(f"  without LLM in either  :  acc {n0['acc']-b0['acc']:+.4f}"
          f"   auc {n0['auc']-b0['auc']:+.4f}   f1 {n0['f1']-b0['f1']:+.4f}")

    # --- squad level ---
    rule("SQUAD LEVEL  (forecast formation, 11 from ~27)")
    print(f"  {'configuration':<32}{'n':>5}{'constrained':>13}{'>=9/11':>9}{'11/11':>8}")
    print("  " + "-" * 68)
    sq = {}
    for name in configs:
        val["p"] = probs[name]
        r = evaluate_squads(val, templates, fc, "markov")
        sq[name] = r
        print(f"  {name:<32}{r['n']:>5}{100*r['constrained']:>12.2f}%"
              f"{100*r['ge9']:>8.1f}%{100*r['perfect']:>7.1f}%")

    a, c = sq["global tabular + LLM"], sq["+ percentiles + LLM"]
    rule("SUMMARY")
    print(f"  best configuration: {max(sq, key=lambda k: sq[k]['constrained'])}")
    print(f"\n  percentiles vs global (+LLM both):")
    print(f"      constrained {100*a['constrained']:.2f}% -> {100*c['constrained']:.2f}%"
          f"   ({100*(c['constrained']-a['constrained']):+.2f}pp)")
    print(f"      >=9/11      {100*a['ge9']:.1f}% -> {100*c['ge9']:.1f}%"
          f"   ({100*(c['ge9']-a['ge9']):+.1f}pp)")
    print(f"      11/11       {100*a['perfect']:.1f}% -> {100*c['perfect']:.1f}%"
          f"   ({100*(c['perfect']-a['perfect']):+.1f}pp)")
    print(f"\n  X/11 distribution: {dict(sorted(c['dist'].items(), reverse=True))}")

    # importance, to see whether the percentiles are actually used
    from xgboost import XGBClassifier
    cols = list(dict.fromkeys(TABULAR + pct_cols + LLM))
    X = df[cols].copy()
    for cc in cols:
        if not pd.api.types.is_numeric_dtype(X[cc]):
            X[cc] = X[cc].astype("category").cat.codes
    mo = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                       random_state=42).fit(X[df.gameweek < cut],
                                            df.is_starter[df.gameweek < cut])
    imp = pd.Series(mo.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    imp = 100 * imp / imp.sum()
    rule("TOP 15 BY GAIN  (percentile features marked)")
    for k, v in imp.head(15).items():
        print(f"  {'PCT ' if k.startswith('pct_') else '    '}{k:<38}{v:>7.2f}%")
    print(f"\n  percentile features take {imp[[i for i in imp.index if i.startswith('pct_')]].sum():.1f}% "
          f"of total gain")


if __name__ == "__main__":
    main()
