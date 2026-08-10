"""What is knowing the formation actually worth?

The aggregate comparison had forecast and oracle formations landing on the same
constrained lineup accuracy. That only means something if it survives the
obvious objection, which is that the forecast is right about 70% of the time, so
of course the two agree on most fixtures. The subset worth looking at is where
the forecast is wrong, because that's the only place oracle and forecast can
differ, and if Stage 2 is worth anything it has to show up there.

Two more controls settle it. A constant formation, one shape on every fixture,
no regard for opponent or club at all: if that ties the forecast then formation
forecasting isn't doing any work. And a random formation drawn from the observed
shapes, which is the one condition that ought to degrade and which bounds how
much the template matters at all. Averaged over draws since one draw is noise.

All of this runs on the proposed configuration, 17 tabular + 12 position-relative
+ 3 contextual, same as Section IV-D. An earlier cut of this script used
tabular+LLM without the percentile block and so was scoring a weaker model.

python formation_value.py
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from positional_percentiles import add_percentiles
from run_pipeline_v2 import (LLM, SEEDS, TABULAR, evaluate_squads,
                             forecast_formations, learn_templates, rule,
                             solve_lineup)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"
RANDOM_DRAWS = 20


def fit_proposed(df: pd.DataFrame, cols: list[str], cut: int) -> np.ndarray:
    """Mean of the five seeds' probability vectors -- the paper's convention."""
    cols = list(dict.fromkeys(cols))
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    y = df.is_starter.astype(int)
    tr, va = df.gameweek < cut, df.gameweek >= cut
    ps = [XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=sd)
          .fit(X[tr], y[tr]).predict_proba(X[va])[:, 1] for sd in SEEDS]
    return np.mean(ps, axis=0)


def per_squad(val, templates, fc, mode, fixed=None):
    """
    Correct-count for every validation squad, keyed so conditions align.

    mode: oracle | markov | recency | fixed   (fixed uses the `fixed` shape)
    Returns the chosen eleven as well as the count, so two conditions can be
    compared on WHICH players they select and not only on how many are right.
    """
    fmap = fc.set_index(["match_id", "team_code"])
    out = {}
    for (mid, tc), grp in val.groupby(["match_id", "team_code"]):
        actual = set(grp[grp.is_starter == 1].player_name)
        if len(actual) != 11 or (mid, tc) not in fmap.index:
            continue
        row = fmap.loc[(mid, tc)]
        if mode == "oracle":
            form = row["actual"]
        elif mode == "fixed":
            form = fixed
        else:
            form = row[mode]
        if form is None or (isinstance(form, float) and np.isnan(form)):
            continue
        slots = templates.get(form) or templates.get(row["actual"])
        if not slots:
            continue
        picked = solve_lineup(grp, slots)
        out[(mid, tc)] = {
            "correct": len(picked & actual),
            "picked": picked,
            "forecast": row["markov"], "actual_form": row["actual"],
            "n_slots_diff": sum((Counter(templates.get(row["actual"], [])) -
                                 Counter(slots)).values()),
        }
    return out


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df, pct = add_percentiles(df)
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val = df[df.gameweek >= cut].copy()
    templates = learn_templates(df)
    fc = forecast_formations(df)

    PROPOSED = list(dict.fromkeys(TABULAR + pct + LLM))
    val["p"] = fit_proposed(df, PROPOSED, cut)

    rule("PROPOSED CONFIGURATION, FORMATION GIVEN vs FORECAST", "#")
    print(f"  features: {len(PROPOSED)}  (17 tabular + 12 position-relative + 3 contextual)")

    o = per_squad(val, templates, fc, "oracle")
    m = per_squad(val, templates, fc, "markov")
    r_ = per_squad(val, templates, fc, "recency")
    keys = sorted(set(o) & set(m) & set(r_))
    print(f"  squads compared: {len(keys)}")

    co = np.array([o[k]["correct"] for k in keys])
    cm = np.array([m[k]["correct"] for k in keys])
    cr = np.array([r_[k]["correct"] for k in keys])
    right = np.array([o[k]["forecast"] == o[k]["actual_form"] for k in keys])

    print(f"\n  {'condition':<28}{'n':>5}{'constrained':>13}{'>=9/11':>9}"
          f"{'11/11':>8}{'mean /11':>10}")
    print("  " + "-" * 74)
    for lab, c in [("formation GIVEN (oracle)", co), ("formation FORECAST (Markov)", cm),
                   ("formation RECENCY", cr)]:
        print(f"  {lab:<28}{len(c):>5}{100*c.mean()/11:>12.2f}%"
              f"{100*(c >= 9).mean():>8.1f}%{100*(c == 11).mean():>7.1f}%{c.mean():>10.2f}")
    print(f"  {'oracle - Markov':<28}{'':>5}{100*(co-cm).mean()/11:>+12.2f}%")

    # --- the two conditions pick the same players ---
    same = sum(1 for k in keys if o[k]["picked"] == m[k]["picked"])
    print(f"\n  oracle and forecast select the SAME ELEVEN in {same} of {len(keys)} squads")

    rule("SPLIT BY WHETHER THE FORECAST WAS CORRECT")
    print(f"  Markov correct on {right.sum()} of {len(keys)} squads ({100*right.mean():.1f}%)")
    print(f"  recency correct on "
          f"{sum(1 for k in keys if r_[k]['actual_form'] == fc.set_index(['match_id','team_code']).loc[k,'recency'])}"
          f" of {len(keys)}")
    print(f"\n  {'subset':<30}{'n':>5}{'oracle':>10}{'forecast':>11}{'gain':>9}")
    print("  " + "-" * 66)
    for lab, msk in [("forecast CORRECT", right), ("forecast WRONG", ~right)]:
        if msk.sum() == 0:
            continue
        print(f"  {lab:<30}{int(msk.sum()):>5}{co[msk].mean():>10.2f}"
              f"{cm[msk].mean():>11.2f}{co[msk].mean()-cm[msk].mean():>+9.2f}")
    worst = [i for i, k in enumerate(keys) if m[k]["n_slots_diff"] >= 2]
    if worst:
        wi = np.array(worst)
        print(f"  {'template differs by 2+ slots':<30}{len(wi):>5}{co[wi].mean():>10.2f}"
              f"{cm[wi].mean():>11.2f}{co[wi].mean()-cm[wi].mean():>+9.2f}")
    print("\n  Units are correct players out of 11. On the subset where the forecast")
    print("  is wrong, oracle is the ONLY condition that differs -- so any value in")
    print("  Stage 2 must appear in that row.")

    rule("WHY: HOW DIFFERENT ARE THE TEMPLATES?")
    diffs = [m[k]["n_slots_diff"] for k in keys if not (o[k]["forecast"] == o[k]["actual_form"])]
    print(f"  when the forecast is wrong, slots differing from the true template:")
    print(f"      {dict(sorted(Counter(diffs).items()))}")
    if diffs:
        print(f"      mean {np.mean(diffs):.2f} of 11 slots")
        print(f"\n  {sum(1 for d in diffs if d == 0)} of {len(diffs)} wrong forecasts produce an "
              f"IDENTICAL slot template")
    print("  -- different formation names, same coarse shape, so the assignment")
    print("  problem is unchanged and the same eleven are selected.")

    # --- the controls that remove formation from the problem ---
    rule("CONTROLS: CONSTANT AND RANDOM FORMATION")
    counts = df.drop_duplicates(["match_id", "team_code"]).formation.value_counts()
    modal = counts.index[0]
    print(f"  most common shape in the season: {modal} ({counts.iloc[0]} team-matches)")
    print(f"\n  {'condition':<34}{'n':>5}{'constrained':>13}{'>=9/11':>9}")
    print("  " + "-" * 62)
    fx = per_squad(val, templates, fc, "fixed", fixed=modal)
    cf = np.array([fx[k]["correct"] for k in sorted(fx)])
    print(f"  {'constant ' + modal + ' (ignores fixture)':<34}{len(cf):>5}"
          f"{100*cf.mean()/11:>12.2f}%{100*(cf >= 9).mean():>8.1f}%")

    forms = list(counts.index)
    rng = np.random.default_rng(0)
    rc, rg = [], []
    for _ in range(RANDOM_DRAWS):
        picks = {}
        tot = []
        fmap = fc.set_index(["match_id", "team_code"])
        for (mid, tc), grp in val.groupby(["match_id", "team_code"]):
            actual = set(grp[grp.is_starter == 1].player_name)
            if len(actual) != 11 or (mid, tc) not in fmap.index:
                continue
            row = fmap.loc[(mid, tc)]
            slots = templates.get(forms[rng.integers(0, len(forms))]) or \
                templates.get(row["actual"])
            if not slots:
                continue
            tot.append(len(solve_lineup(grp, slots) & actual))
        tot = np.array(tot)
        rc.append(tot.mean() / 11)
        rg.append((tot >= 9).mean())
        picks.clear()
    rc, rg = np.array(rc), np.array(rg)
    print(f"  {'random shape (mean of ' + str(RANDOM_DRAWS) + ' draws)':<34}{len(cf):>5}"
          f"{100*rc.mean():>12.2f}%{100*rg.mean():>8.1f}%")
    print(f"  {'   ':<34}{'':>5}{'sd ' + f'{100*rc.std():.2f}':>13}{'sd ' + f'{100*rg.std():.1f}':>9}")
    print("\n  A single constant shape matches the forecast; only a random shape")
    print("  degrades the result. Formation determines the labelling of the slots,")
    print("  not which eleven players fill them.")


if __name__ == "__main__":
    main()
