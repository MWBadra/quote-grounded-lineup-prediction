"""Is the headline number stable across chronological folds, and does it survive
the two baselines a reviewer is going to reach for first?

  - trivial baseline: sort by recency, no model at all, same assignment stage.
    v1 never reported one and beat it by under two points. Better we find that
    than a referee does.
  - architecture comparison: if logistic regression keeps up with boosting then
    the ceiling is the feature space, not the algorithm, and we should say so
    instead of letting the reader assume XGBoost is doing the work.

Folds are forward splits, train on everything up to gameweek k and test the
next block, so no fold ever sees its own future.

python cross_validation.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from positional_percentiles import add_percentiles
from run_pipeline_v2 import (LLM, TABULAR, evaluate_squads, forecast_formations,
                             learn_templates, rule)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"
SEEDS = (1, 2, 3, 4, 5)


def encode(df, cols, sentinel: bool = False):
    """
    sentinel=False keeps missing values missing, which is how the proposed model
    is fitted everywhere else in the paper -- roughly ninety per cent of the
    contextual block is absent by construction and XGBoost learns a default
    direction for it. sentinel=True is used ONLY for the architecture comparison,
    where logistic regression and random forests cannot accept a NaN and all four
    arms must therefore see identical inputs.
    """
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X.fillna(-999) if sentinel else X


def folds(gws: list[int], n: int = 5):
    """Forward chains: each fold trains on everything before its test block."""
    start = int(len(gws) * 0.45)          # first fold needs enough history
    edges = np.linspace(start, len(gws), n + 1).astype(int)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        yield gws[:lo], gws[lo:hi]


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df, pct = add_percentiles(df)
    BEST = list(dict.fromkeys(TABULAR + pct + LLM))
    NOLLM = list(dict.fromkeys(TABULAR + pct))
    gws = sorted(df.gameweek.unique())
    templates, fc = learn_templates(df), forecast_formations(df)

    # --- 1. chronological cross-validation ---
    rule("1. FORWARD-CHAINING CROSS-VALIDATION", "#")
    print(f"  {'fold':<6}{'train GW':<12}{'test GW':<12}{'n test':>8}"
          f"{'acc':>9}{'ROC-AUC':>10}{'F1':>9}{'constrained':>13}{'>=9/11':>9}")
    print("  " + "-" * 92)
    rows = []
    for k, (tr_gw, te_gw) in enumerate(folds(gws), 1):
        tr, te = df.gameweek.isin(tr_gw), df.gameweek.isin(te_gw)
        if te.sum() < 200:
            continue
        X, y = encode(df, BEST), df.is_starter.astype(int)
        ps = []
        for sd in SEEDS:
            m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", random_state=sd).fit(X[tr], y[tr])
            ps.append(m.predict_proba(X[te])[:, 1])
        p = np.mean(ps, axis=0)
        acc, auc, f1 = (accuracy_score(y[te], p >= .5), roc_auc_score(y[te], p),
                        f1_score(y[te], p >= .5))
        sub = df[te].copy()
        sub["p"] = p
        r = evaluate_squads(sub, templates, fc, "markov")
        rows.append((acc, auc, f1, r["constrained"], r["ge9"]))
        print(f"  {k:<6}{f'{tr_gw[0]}-{tr_gw[-1]}':<12}{f'{te_gw[0]}-{te_gw[-1]}':<12}"
              f"{int(te.sum()):>8}{acc:>9.4f}{auc:>10.4f}{f1:>9.4f}"
              f"{100*r['constrained']:>12.2f}%{100*r['ge9']:>8.1f}%")
    a = np.array(rows)
    print("  " + "-" * 92)
    print(f"  {'mean':<30}{'':<12}{'':>8}{a[:,0].mean():>9.4f}{a[:,1].mean():>10.4f}"
          f"{a[:,2].mean():>9.4f}{100*a[:,3].mean():>12.2f}%{100*a[:,4].mean():>8.1f}%")
    print(f"  {'std':<30}{'':<12}{'':>8}{a[:,0].std():>9.4f}{a[:,1].std():>10.4f}"
          f"{a[:,2].std():>9.4f}{100*a[:,3].std():>12.2f}%{100*a[:,4].std():>8.1f}%")

    # --- 2. trivial baselines ---
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()
    rule("2. TRIVIAL BASELINES  (no model, same assignment stage)")
    print(f"  {'method':<44}{'constrained':>13}{'>=9/11':>9}{'11/11':>8}")
    print("  " + "-" * 76)
    trivia = {
        "start_rate_last_5 only":              val.start_rate_last_5,
        "start_rate + consecutive_starts":     val.start_rate_last_5 + 0.01 * val.consecutive_starts,
        "consecutive_starts only":             val.consecutive_starts,
        "availability_score only":             val.availability_score.fillna(0),
        "random":                              pd.Series(
            np.random.default_rng(0).random(len(val)), index=val.index),
    }
    for name, series in trivia.items():
        val["p"] = series.to_numpy(dtype=float)
        r = evaluate_squads(val, templates, fc, "markov")
        print(f"  {name:<44}{100*r['constrained']:>12.2f}%"
              f"{100*r['ge9']:>8.1f}%{100*r['perfect']:>7.1f}%")

    X, y = encode(df, BEST), df.is_starter.astype(int)
    tr, te = df.gameweek < cut, df.gameweek >= cut
    ps = [XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                        random_state=sd).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
          for sd in SEEDS]
    val["p"] = np.mean(ps, axis=0)
    full = evaluate_squads(val, templates, fc, "markov")
    print("  " + "-" * 76)
    print(f"  {'FULL PIPELINE (tabular + pct + LLM)':<44}{100*full['constrained']:>12.2f}%"
          f"{100*full['ge9']:>8.1f}%{100*full['perfect']:>7.1f}%")

    val["p"] = val.start_rate_last_5.to_numpy(dtype=float)
    triv = evaluate_squads(val, templates, fc, "markov")
    print(f"\n  gain of the full pipeline over the best trivial rule: "
          f"{100*(full['constrained']-triv['constrained']):+.2f}pp constrained, "
          f"{100*(full['ge9']-triv['ge9']):+.1f}pp at >=9/11")

    # --- 3. architecture comparison ---
    rule("3. ARCHITECTURE COMPARISON  (identical features and split)")
    print(f"  {'model':<40}{'accuracy':>10}{'ROC-AUC':>10}{'F1':>9}"
          f"{'constrained':>13}{'>=9/11':>9}")
    print("  " + "-" * 92)
    # Sentinel imputation from here down, so that all four architectures see
    # identical inputs; this is why the absolute values below sit slightly under
    # the headline table.
    X = encode(df, BEST, sentinel=True)
    archs = {
        "XGBoost (proposed)": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, random_state=42, n_jobs=-1),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
    }
    try:
        from lightgbm import LGBMClassifier
        archs["LightGBM"] = LGBMClassifier(n_estimators=300, max_depth=5,
                                           learning_rate=0.05, random_state=42,
                                           verbose=-1)
    except ImportError:
        pass
    for name, mo in archs.items():
        mo.fit(X[tr], y[tr])
        p = mo.predict_proba(X[te])[:, 1]
        val["p"] = p
        r = evaluate_squads(val, templates, fc, "markov")
        print(f"  {name:<40}{accuracy_score(y[te], p>=.5):>10.4f}"
              f"{roc_auc_score(y[te], p):>10.4f}{f1_score(y[te], p>=.5):>9.4f}"
              f"{100*r['constrained']:>12.2f}%{100*r['ge9']:>8.1f}%")


if __name__ == "__main__":
    main()
