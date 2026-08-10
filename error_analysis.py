"""Where does the tabular model actually fail?

The pooled contextual increment is small, which is only interesting if the
errors it would have to fix sit somewhere identifiable. So: fit the 17-feature
tabular model under the paper's protocol, then split the validation errors by
start_rate_last_5 (the strongest single tabular feature) and pull out the two
failure modes the argument rests on, regulars who were rested and fringe
players who were recalled.

Second half checks whether the extracted labels reach that same population.
Start rates there are quoted against the base rate on covered rows, not the
all-row base rate; the latter would flatter the labels, since pre-match copy
mostly talks about first-teamers anyway.

python error_analysis.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from run_pipeline_v2 import SEEDS, TABULAR, rule

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"

ROTATION_NAME = {0.0: "confirmed starter", 0.2: "expected starter",
                 0.7: "competition lost", 0.8: "rotation risk",
                 1.0: "ruled out"}


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    y = df.is_starter.astype(int)
    tr, va = df.gameweek < cut, df.gameweek >= cut

    X = df[TABULAR].copy()
    for c in TABULAR:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    p = np.mean([XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=sd)
                 .fit(X[tr], y[tr]).predict_proba(X[va])[:, 1] for sd in SEEDS], axis=0)

    v = df[va].copy()
    v["pred"] = (p >= .5).astype(int)
    v["err"] = (v.pred != v.is_starter).astype(int)
    E = v[v.err == 1]
    sr = v.start_rate_last_5

    rule("WHERE THE TABULAR MODEL FAILS", "#")
    print(f"  tabular model: {len(TABULAR)} features, {len(SEEDS)} seeds, mean probability vector")
    print(f"  validation rows {len(v):,} | errors {len(E):,} | error rate {v.err.mean():.4f}")

    rule("ERROR RATE BY RECENT START RATE")
    print(f"  {'stratum':<34}{'n':>8}{'error rate':>13}")
    print("  " + "-" * 56)
    for lab, msk in [("started none of the last 5", sr == 0),
                     ("rotation middle ground (0 < r < 1)", (sr > 0) & (sr < 1)),
                     ("started all of the last 5", sr == 1)]:
        print(f"  {lab:<34}{int(msk.sum()):>8}{100*v[msk].err.mean():>12.1f}%")
    mid = ((E.start_rate_last_5 > 0) & (E.start_rate_last_5 < 1)).mean()
    print(f"\n  share of ALL errors falling in the rotation middle ground: {100*mid:.1f}%")

    rule("WHAT THE ERRORS ARE")
    print(f"  {'failure mode':<44}{'share of errors':>17}")
    print("  " + "-" * 63)
    for lab, msk in [
            ("established regular rested (r = 1, did not start)",
             (E.start_rate_last_5 == 1) & (E.is_starter == 0)),
            ("fringe player recalled (r = 0, started)",
             (E.start_rate_last_5 == 0) & (E.is_starter == 1)),
            ("new to the league", E.is_new_to_league == 1)]:
        print(f"  {lab:<44}{100*msk.mean():>16.1f}%")

    rule("DO THE EXTRACTED LABELS REACH THAT POPULATION?")
    g = df[df.llm_player_rotation.notna()]
    print(f"  rows carrying a signal: {len(g):,} of {len(df):,} ({100*len(g)/len(df):.2f}%)")
    print(f"  base start rate among them {g.is_starter.mean():.3f}   "
          f"across all rows {df.is_starter.mean():.3f}")
    print(f"\n  {'label':<22}{'n':>7}{'start rate':>13}")
    print("  " + "-" * 44)
    for val_, name in sorted(ROTATION_NAME.items()):
        s = g[g.llm_player_rotation == val_]
        if len(s):
            print(f"  {name:<22}{len(s):>7}{100*s.is_starter.mean():>12.1f}%")
    print("\n  The comparator is the covered-row base rate, not the all-row one:")
    print("  reporting against the latter would credit the labels with a separation")
    print("  that is partly just which players pre-match copy chooses to discuss.")


if __name__ == "__main__":
    main()
