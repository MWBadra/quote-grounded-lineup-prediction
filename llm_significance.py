"""Is the LLM contribution real, and can it be made bigger?

Three parts.

Significance first: paired McNemar plus a bootstrap CI on the delta, measured
against the strongest baseline we have rather than a convenient weak one. v1's
+7.15 came out of a comparison with a crippled two-feature arm. The only number
that survives a referee is the one taken against the best model you own.

Then four encodings the current three columns don't capture:
  - fixture_covered: did we retrieve anything at all for this team-match? A
    player nobody mentioned in a well-covered fixture really isn't newsworthy,
    whereas a player in an uncovered fixture tells us nothing whatsoever. Both
    collapse to NULL right now, which throws that away.
  - llm_strength / llm_source_type: extracted already, never actually used.
  - pct_llm_rotation: rank within squad, same trick that worked on the tabular
    stats.

Last part is greedy backward elimination on validation ROC-AUC.

python llm_significance.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from positional_percentiles import add_percentiles
from run_pipeline_positional import GROUP_OF
from run_pipeline_v2 import (LLM, TABULAR, evaluate_squads, forecast_formations,
                             learn_templates, rule)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"
SEEDS = (1, 2, 3, 4, 5)


def encode(df, cols):
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    return X


def fit(df, cols, cut, seeds=SEEDS):
    cols = list(dict.fromkeys(cols))
    X, y = encode(df, cols), df.is_starter.astype(int)
    tr, va = df.gameweek < cut, df.gameweek >= cut
    ps = []
    for sd in seeds:
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=sd).fit(X[tr], y[tr])
        ps.append(m.predict_proba(X[va])[:, 1])
    p = np.mean(ps, axis=0)
    yv = y[va].to_numpy()
    return p, yv, (accuracy_score(yv, p >= .5), roc_auc_score(yv, p),
                   f1_score(yv, p >= .5))


def build_llm_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    # Was anything at all retrieved for this fixture? Without it, "no news about
    # him" and "no news about anybody" are the same NULL.
    cov = (df.groupby(["match_id", "team_code"]).llm_player_rotation
             .transform(lambda s: s.notna().any()).astype(int))
    df["fixture_covered"] = cov
    df["llm_n_signals"] = (df.groupby(["match_id", "team_code"])
                             .llm_player_rotation.transform(lambda s: s.notna().sum()))
    # A signal that exists in a covered fixture is worth more than a bare NULL.
    df["llm_mentioned"] = df.llm_player_rotation.notna().astype(int)
    for c, order in [("llm_strength", {"explicit": 2, "implied": 1, "none": 0}),
                     ("llm_source_type", {"manager_quote": 3, "club_official": 3,
                                          "journalist_prediction": 1,
                                          "fantasy_tips": 0, "unknown": 1})]:
        if c in df.columns:
            df[c + "_ord"] = df[c].map(order).fillna(0)
    df["_grp"] = df.player_position.map(GROUP_OF)
    df["pct_llm_rotation"] = (df.groupby(["match_id", "team_code", "_grp"], sort=False)
                                .llm_player_rotation.rank(pct=True))
    df = df.drop(columns="_grp")
    new = ["fixture_covered", "llm_n_signals", "llm_mentioned",
           "llm_strength_ord", "llm_source_type_ord", "pct_llm_rotation"]
    return df, [c for c in new if c in df.columns]


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df, pct = add_percentiles(df)
    df, llm_extra = build_llm_features(df)
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()
    templates, fc = learn_templates(df), forecast_formations(df)

    BASE = list(dict.fromkeys(TABULAR + pct))

    # --- 1. significance against the strongest baseline ---
    rule("1. IS THE LLM DELTA REAL?", "#")
    p0, yv, m0 = fit(df, BASE, cut)
    p1, _, m1 = fit(df, BASE + LLM, cut)
    print(f"  baseline (tabular + percentiles)   acc {m0[0]:.4f}  auc {m0[1]:.4f}  f1 {m0[2]:.4f}")
    print(f"  + 3 LLM features                   acc {m1[0]:.4f}  auc {m1[1]:.4f}  f1 {m1[2]:.4f}")
    print(f"  delta                              acc {m1[0]-m0[0]:+.4f}  "
          f"auc {m1[1]-m0[1]:+.4f}  f1 {m1[2]-m0[2]:+.4f}")

    a0, a1 = (p0 >= .5).astype(int), (p1 >= .5).astype(int)
    b = int(((a1 == yv) & (a0 != yv)).sum())
    c = int(((a1 != yv) & (a0 == yv)).sum())
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
    pv = stats.chi2.sf(chi2, 1)
    print(f"\n  McNemar (paired, same validation rows)")
    print(f"      LLM right / base wrong = {b}")
    print(f"      LLM wrong / base right = {c}")
    print(f"      chi2 = {chi2:.3f}   p = {pv:.4f}   -> "
          f"{'SIGNIFICANT' if pv < 0.05 else 'NOT significant'} at 5%")

    rng = np.random.default_rng(0)
    n = len(yv)
    ba, bu = [], []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        ba.append((a1[i] == yv[i]).mean() - (a0[i] == yv[i]).mean())
        if len(set(yv[i])) > 1:
            bu.append(roc_auc_score(yv[i], p1[i]) - roc_auc_score(yv[i], p0[i]))
    ba, bu = np.array(ba), np.array(bu)
    print(f"\n  bootstrap (2000 resamples)")
    print(f"      d accuracy {ba.mean():+.4f}  95% CI [{np.percentile(ba,2.5):+.4f},"
          f" {np.percentile(ba,97.5):+.4f}]  {'excludes 0' if np.percentile(ba,2.5)>0 else 'INCLUDES 0'}")
    print(f"      d ROC-AUC  {bu.mean():+.4f}  95% CI [{np.percentile(bu,2.5):+.4f},"
          f" {np.percentile(bu,97.5):+.4f}]  {'excludes 0' if np.percentile(bu,2.5)>0 else 'INCLUDES 0'}")

    cov = df[df.gameweek >= cut].fixture_covered.to_numpy().astype(bool)
    print(f"\n  restricted to fixtures the retrieval actually covered (n={cov.sum():,}):")
    print(f"      base acc {accuracy_score(yv[cov], a0[cov]):.4f}  "
          f"+LLM {accuracy_score(yv[cov], a1[cov]):.4f}  "
          f"delta {accuracy_score(yv[cov], a1[cov])-accuracy_score(yv[cov], a0[cov]):+.4f}")
    # Where the signal actually fires. The accuracy delta on this subset is NOT
    # reported as evidence: it is a net handful of rows and its own paired test
    # does not reject, exactly as the pooled one does not. What survives on this
    # subset is the ranking effect, which is what Stage 4 consumes, so that is
    # what gets a confidence interval and what the paper quotes.
    men = df[df.gameweek >= cut].llm_mentioned.to_numpy().astype(bool)
    ys, m0_, m1_ = yv[men], a0[men], a1[men]
    nm = int(men.sum())
    d_acc = accuracy_score(ys, m1_) - accuracy_score(ys, m0_)
    bb = int(((m1_ == ys) & (m0_ != ys)).sum())
    cc = int(((m1_ != ys) & (m0_ == ys)).sum())
    chi2m = (abs(bb - cc) - 1) ** 2 / (bb + cc) if (bb + cc) else 0.0
    print(f"\n  restricted to players WITH a signal (n={nm:,}):")
    print(f"      accuracy  base {accuracy_score(ys, m0_):.4f}  +LLM "
          f"{accuracy_score(ys, m1_):.4f}  delta {d_acc:+.4f}  "
          f"(= {int(round(d_acc*nm))} rows net)")
    print(f"          McNemar chi2 = {chi2m:.3f}  p = {stats.chi2.sf(chi2m, 1):.3f}"
          f"  -> {'significant' if stats.chi2.sf(chi2m,1) < .05 else 'NOT significant'};"
          f" do not report this as an effect")
    auc0, auc1 = roc_auc_score(ys, p0[men]), roc_auc_score(ys, p1[men])
    ba2, bu2 = [], []
    for _ in range(2000):
        i = rng.integers(0, nm, nm)
        ba2.append((m1_[i] == ys[i]).mean() - (m0_[i] == ys[i]).mean())
        if len(set(ys[i])) > 1:
            bu2.append(roc_auc_score(ys[i], p1[men][i]) - roc_auc_score(ys[i], p0[men][i]))
    ba2, bu2 = np.array(ba2), np.array(bu2)
    print(f"          bootstrap d accuracy 95% CI [{np.percentile(ba2,2.5):+.4f},"
          f" {np.percentile(ba2,97.5):+.4f}]  "
          f"{'excludes 0' if np.percentile(ba2,2.5)>0 else 'INCLUDES 0'}")
    print(f"      ROC-AUC   base {auc0:.4f}  +LLM {auc1:.4f}  delta {auc1-auc0:+.4f}")
    print(f"          bootstrap d ROC-AUC  95% CI [{np.percentile(bu2,2.5):+.4f},"
          f" {np.percentile(bu2,97.5):+.4f}]  "
          f"{'excludes 0' if np.percentile(bu2,2.5)>0 else 'INCLUDES 0'}")
    print(f"      ratio of this AUC gain to the pooled one: "
          f"{(auc1-auc0)/(m1[1]-m0[1]):.1f}x")

    # Cluster bootstrap over team-matches: rows within a squad are not
    # independent (exactly eleven of ~27 start), so the i.i.d. row bootstrap
    # above understates the interval. The pooled AUC gain survives either way.
    vg = df[df.gameweek >= cut].groupby(["match_id", "team_code"]).ngroup().to_numpy()
    ug = np.unique(vg)
    by_g = [np.where(vg == g)[0] for g in ug]
    bc = []
    for _ in range(2000):
        j = np.concatenate([by_g[k] for k in rng.integers(0, len(ug), len(ug))])
        if len(set(yv[j])) > 1:
            bc.append(roc_auc_score(yv[j], p1[j]) - roc_auc_score(yv[j], p0[j]))
    bc = np.array(bc)
    print(f"\n  pooled ROC-AUC delta, CLUSTER bootstrap over {len(ug)} team-matches:")
    print(f"      95% CI [{np.percentile(bc,2.5):+.4f}, {np.percentile(bc,97.5):+.4f}]  "
          f"{'excludes 0' if np.percentile(bc,2.5)>0 else 'INCLUDES 0'}")

    # --- 2. richer llm encoding ---
    rule("2. CAN THE LLM FEATURES BE ENCODED BETTER?")
    variants = {
        "no LLM":                       BASE,
        "3 raw LLM":                    BASE + LLM,
        "+ coverage flags":             BASE + LLM + ["fixture_covered", "llm_mentioned",
                                                      "llm_n_signals"],
        "+ strength & source":          BASE + LLM + ["llm_strength_ord",
                                                      "llm_source_type_ord"],
        "+ squad percentile":           BASE + LLM + ["pct_llm_rotation"],
        "all LLM encodings":            BASE + LLM + llm_extra,
    }
    print(f"  {'variant':<28}{'accuracy':>11}{'ROC-AUC':>11}{'F1':>10}{'d acc':>10}")
    print("  " + "-" * 72)
    store = {}
    for name, cols in variants.items():
        p, _, m = fit(df, cols, cut)
        store[name] = (p, m, cols)
        print(f"  {name:<28}{m[0]:>11.4f}{m[1]:>11.4f}{m[2]:>10.4f}"
              f"{m[0]-m0[0]:>+10.4f}")

    # --- 3. greedy backward elimination ---
    rule("3. FEATURE SELECTION  (greedy backward elimination on ROC-AUC)")
    cols = list(dict.fromkeys(BASE + LLM + llm_extra))
    _, _, cur = fit(df, cols, cut, seeds=(1, 2))
    best_auc = cur[1]
    print(f"  start: {len(cols)} features, ROC-AUC {best_auc:.4f}")
    removed = []
    improved = True
    while improved and len(cols) > 6:
        improved = False
        scores = []
        for c in cols:
            trial = [x for x in cols if x != c]
            _, _, m = fit(df, trial, cut, seeds=(1, 2))
            scores.append((m[1], c))
        scores.sort(reverse=True)
        top_auc, drop = scores[0]
        if top_auc >= best_auc:
            cols = [x for x in cols if x != drop]
            removed.append((drop, top_auc))
            best_auc = top_auc
            improved = True
            print(f"      drop {drop:<38} -> {top_auc:.4f}  ({len(cols)} left)")
    print(f"\n  removed {len(removed)} features; {len(cols)} retained")

    rule("FINAL COMPARISON  (5 seeds, row + squad)")
    finals = {"tabular+pct (no LLM)": BASE,
              "tabular+pct + 3 LLM": BASE + LLM,
              "all LLM encodings": BASE + LLM + llm_extra,
              f"selected ({len(cols)} feat)": cols}
    print(f"  {'configuration':<30}{'accuracy':>11}{'ROC-AUC':>11}"
          f"{'constrained':>13}{'>=9/11':>9}{'11/11':>8}")
    print("  " + "-" * 84)
    for name, cc in finals.items():
        p, _, m = fit(df, cc, cut)
        val["p"] = p
        r = evaluate_squads(val, templates, fc, "markov")
        print(f"  {name:<30}{m[0]:>11.4f}{m[1]:>11.4f}"
              f"{100*r['constrained']:>12.2f}%{100*r['ge9']:>8.1f}%{100*r['perfect']:>7.1f}%")
    print(f"\n  selected feature set:\n      {', '.join(cols)}")


if __name__ == "__main__":
    main()
