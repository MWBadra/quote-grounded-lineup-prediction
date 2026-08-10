"""What do the monotone constraints actually cost us?

The old pipeline constrained every feature with an obvious direction and
reported numbers from that constrained fit. This checks whether that was worth
doing: same proposed configuration fitted twice, with and without, five seeds
each, metric taken on the mean probability vector (the same one Stage 4 eats).

Careful with which columns get constrained. "Recency" is four columns, not two,
since the raw start rate and consecutive-start run both have percentile copies.
Constrain only the raw pair and you get a much rosier answer, because the model
just routes the same information through the unconstrained copies. Hence the
seven-column spec below; the narrower one is kept at the bottom as a sanity
check.

python monotone_ablation.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

from positional_percentiles import add_percentiles
from run_pipeline_v2 import (LLM, SEEDS, TABULAR, evaluate_squads,
                             forecast_formations, learn_templates, rule)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"

# +1 = more of it should never push the starting probability down.
# -1 = rotation runs 0 (confirmed starter) to 1 (ruled out), so it's inverted.
DIRECTION = {
    "start_rate_last_5":       +1,
    "consecutive_starts":      +1,
    "pct_start_rate_last_5":   +1,
    "pct_consecutive_starts":  +1,
    "availability_score":      +1,
    "llm_player_rotation":     -1,
    "llm_team_rotation":       -1,
}


def fit(df, cols, cut, constraints=None):
    cols = list(dict.fromkeys(cols))
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    y = df.is_starter.astype(int)
    tr, va = df.gameweek < cut, df.gameweek >= cut
    kw = {} if constraints is None else {"monotone_constraints": constraints}
    ps = [XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=sd, **kw)
          .fit(X[tr], y[tr]).predict_proba(X[va])[:, 1] for sd in SEEDS]
    return np.mean(ps, axis=0), y[va].to_numpy()


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df, pct = add_percentiles(df)
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()
    templates, fc = learn_templates(df), forecast_formations(df)

    PROPOSED = list(dict.fromkeys(TABULAR + pct + LLM))
    cons = tuple(DIRECTION.get(c, 0) for c in PROPOSED)

    rule("MONOTONE CONSTRAINTS ON THE PROPOSED CONFIGURATION", "#")
    print(f"  {len(PROPOSED)} features, of which {sum(1 for v in cons if v)} constrained:")
    for c in PROPOSED:
        if c in DIRECTION:
            print(f"      {DIRECTION[c]:+d}  {c}")

    out = {}
    for lab, cc in [("unconstrained", None), ("monotone", cons)]:
        p, yv = fit(df, PROPOSED, cut, cc)
        val["p"] = p
        r = evaluate_squads(val, templates, fc, "markov")
        out[lab] = (accuracy_score(yv, p >= .5), roc_auc_score(yv, p),
                    r["constrained"], r["ge9"])

    print(f"\n  {'fit':<18}{'accuracy':>11}{'ROC-AUC':>11}{'constrained':>14}{'>=9/11':>9}")
    print("  " + "-" * 64)
    for lab in ("unconstrained", "monotone"):
        a, u, c, g = out[lab]
        print(f"  {lab:<18}{a:>11.4f}{u:>11.4f}{100*c:>13.2f}%{100*g:>8.1f}%")

    u_, m_ = out["unconstrained"], out["monotone"]
    rule("COST OF IMPOSING THE CONSTRAINTS")
    print(f"  constrained lineup accuracy   {100*(u_[2]-m_[2]):>+7.2f} pp")
    print(f"  lineups >= 9/11               {100*(u_[3]-m_[3]):>+7.1f} pp")
    print(f"  ROC-AUC, absolute             {u_[1]-m_[1]:>+7.4f}")
    print(f"  ROC-AUC, relative             {100*(u_[1]-m_[1])/u_[1]:>+7.2f} %")
    print("\n  The squad-level cost decides the matter: the assignment stage consumes")
    print("  the ranking, and a constrained ranking loses nine-of-eleven lineups.")

    # and the narrow reading, raw recency pair only, for comparison
    narrow = {k: v for k, v in DIRECTION.items() if not k.startswith("pct_")}
    p, yv = fit(df, PROPOSED, cut, tuple(narrow.get(c, 0) for c in PROPOSED))
    val["p"] = p
    r = evaluate_squads(val, templates, fc, "markov")
    rule("SENSITIVITY: CONSTRAINING ONLY THE RAW RECENCY PAIR (5 columns)")
    print(f"  constrained {100*r['constrained']:.2f}%  (cost {100*(u_[2]-r['constrained']):+.2f} pp)"
          f"   >=9/11 {100*r['ge9']:.1f}%  (cost {100*(u_[3]-r['ge9']):+.1f} pp)")
    print("  The percentile recency columns absorb the constrained information, so")
    print("  the narrower specification understates the cost. This is why the seven-")
    print("  column specification above is the one reported.")


if __name__ == "__main__":
    main()
