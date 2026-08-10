"""End-to-end evaluation for v2: player probabilities -> formation -> positional
assignment, plus the feature ablation grid.

Three differences from v1 that actually matter for the paper.

The candidate pool is the ~27-player availability squad, not the ~20-man
matchday squad. v1 was effectively telling the model which 20 the manager had
already settled on. Picking 11 out of 27 is the honest version of the problem
and it scores lower, which is fine.

Formation runs in three conditions, oracle, Markov chain and plain recency, so
the cost of not knowing the shape gets measured instead of assumed. v1 had no
way to separate formation error from selection error.

Positional templates come out of the data rather than being hand-written. For
each formation string take the modal composition of coarse slots among the
players who actually started, so the template matches how clubs really set up in
that shape and not how the diagram says they ought to.

python run_pipeline_v2.py
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset" / "pl_2025_2026_final.csv"
SEEDS = (1, 2, 3, 4, 5)
INELIGIBLE = 1e6

# Which coarse slots a player of each coarse position may fill. Deliberately
# permissive across the wide/centre boundary in the same line, and one line
# adjacent -- a full-back covers centre-back, a winger covers wide midfield --
# because the season's own coordinates show players move that far routinely.
ELIGIBILITY = {
    "GK":       {"GK"},
    "D-CENTRE": {"D-CENTRE", "D-WIDE"},
    "D-WIDE":   {"D-WIDE", "D-CENTRE", "M-WIDE"},
    "M-CENTRE": {"M-CENTRE", "M-WIDE", "D-CENTRE"},
    "M-WIDE":   {"M-WIDE", "M-CENTRE", "F-WIDE", "D-WIDE"},
    "F-CENTRE": {"F-CENTRE", "F-WIDE", "M-CENTRE"},
    "F-WIDE":   {"F-WIDE", "F-CENTRE", "M-WIDE"},
}

TABULAR = ["start_rate_last_5", "consecutive_starts", "availability_score",
           "player_position", "clean_sheets", "recoveries",
           "clearances_blocks_interceptions", "f5_accurate_passes_percent",
           "goals_scored", "assists", "f5_chances_created", "f5_total_shots",
           "f5_fouls_committed", "yellow_cards", "red_cards", "is_new_to_league",
           "gk_saves"]
LLM = ["llm_player_rotation", "llm_manager_sentiment", "llm_team_rotation"]


def rule(t: str, ch: str = "=") -> None:
    print("\n" + ch * 86)
    print(t)
    print(ch * 86, flush=True)


# --- formation templates, learned from the data ---

def learn_templates(df: pd.DataFrame) -> dict[str, list[str]]:
    """formation string -> the 11 coarse slots clubs actually field in it."""
    tpl = {}
    starters = df[df.is_starter == 1]
    for f, grp in starters.groupby("formation"):
        per = grp.groupby(["match_id", "team_code"]).player_position.apply(
            lambda s: tuple(sorted(s)))
        if per.empty:
            continue
        tpl[f] = list(Counter(per).most_common(1)[0][0])
    return tpl


# --- formation forecasting ---

def forecast_formations(df: pd.DataFrame) -> pd.DataFrame:
    """Markov and recency predictions per team-match, using only prior matches."""
    hist = (df.drop_duplicates(["match_id", "team_code"])
              .sort_values(["team_code", "gameweek"])
              [["team_code", "gameweek", "match_id", "formation"]])
    rows = []
    for tc, grp in hist.groupby("team_code"):
        seq = grp.formation.tolist()
        mids = grp.match_id.tolist()
        gws = grp.gameweek.tolist()
        for i in range(len(seq)):
            prior = seq[:i]
            if not prior:
                mk = rc = None
            else:
                rc = prior[-1]
                trans = defaultdict(Counter)
                for a, b in zip(prior, prior[1:]):
                    trans[a][b] += 1
                mk = (trans[rc].most_common(1)[0][0] if trans[rc] else rc)
            rows.append({"team_code": tc, "match_id": mids[i], "gameweek": gws[i],
                         "actual": seq[i], "markov": mk, "recency": rc})
    return pd.DataFrame(rows)


# --- stage 4 ---

def solve_lineup(players: pd.DataFrame, slots: list[str]) -> set[str]:
    """Hungarian assignment of players to the 11 slots, maximising total P."""
    if players.empty or not slots:
        return set()
    cost = np.full((len(players), len(slots)), INELIGIBLE)
    pos = players.player_position.tolist()
    prob = players.p.tolist()
    for i, ppos in enumerate(pos):
        for j, slot in enumerate(slots):
            if ppos in ELIGIBILITY.get(slot, {slot}):
                cost[i, j] = -prob[i]
    r, c = linear_sum_assignment(cost)
    names = players.player_name.tolist()
    return {names[i] for i, j in zip(r, c) if cost[i, j] < INELIGIBLE}


def evaluate_squads(val: pd.DataFrame, templates: dict, fc: pd.DataFrame,
                    mode: str) -> dict:
    """mode: oracle | markov | recency"""
    fmap = fc.set_index(["match_id", "team_code"])
    counts, raws = [], []
    fhit = ftot = 0
    for (mid, tc), grp in val.groupby(["match_id", "team_code"]):
        actual = set(grp[grp.is_starter == 1].player_name)
        if len(actual) != 11:
            continue
        key = (mid, tc)
        if key not in fmap.index:
            continue
        row = fmap.loc[key]
        form = row["actual"] if mode == "oracle" else row[mode]
        if form is None or (isinstance(form, float) and np.isnan(form)):
            continue
        if mode != "oracle":
            ftot += 1
            fhit += int(form == row["actual"])
        slots = templates.get(form) or templates.get(row["actual"])
        if not slots:
            continue
        picked = solve_lineup(grp, slots)
        counts.append(len(picked & actual))
        top11 = set(grp.nlargest(11, "p").player_name)
        raws.append(len(top11 & actual))
    n = len(counts)
    return {
        "n": n,
        "constrained": float(np.mean(counts) / 11) if n else 0.0,
        "raw_top11": float(np.mean(raws) / 11) if n else 0.0,
        "ge9": float(np.mean([c >= 9 for c in counts])) if n else 0.0,
        "perfect": float(np.mean([c == 11 for c in counts])) if n else 0.0,
        "formation_acc": (fhit / ftot) if ftot else None,
        "dist": Counter(counts),
    }


# --- modelling ---

def fit_predict(df: pd.DataFrame, cols: list[str], cut: int) -> tuple[np.ndarray, dict]:
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    y = df.is_starter.astype(int)
    tr, va = df.gameweek < cut, df.gameweek >= cut
    preds, met = [], []
    for sd in SEEDS:
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", random_state=sd).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        preds.append(p)
        met.append((accuracy_score(y[va], p >= .5), roc_auc_score(y[va], p),
                    f1_score(y[va], p >= .5)))
    a = np.array(met)
    return np.mean(preds, axis=0), {
        "acc": a[:, 0].mean(), "acc_sd": a[:, 0].std(),
        "auc": a[:, 1].mean(), "auc_sd": a[:, 1].std(), "f1": a[:, 2].mean()}


def main() -> None:
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    gws = sorted(df.gameweek.unique())
    cut = gws[int(len(gws) * 0.7)]
    val_mask = df.gameweek >= cut

    rule("SETUP", "#")
    print(f"  rows {len(df):,} | team-matches {df.groupby(['match_id','team_code']).ngroups}")
    print(f"  chronological split at GW{cut}: "
          f"train {int((~val_mask).sum()):,} / val {int(val_mask.sum()):,}")
    print(f"  candidate pool mean {df.groupby(['match_id','team_code']).size().mean():.1f} players")
    print(f"  LLM coverage {100*df.llm_player_rotation.notna().mean():.2f}% of rows")

    templates = learn_templates(df)
    print(f"  formation templates learned: {len(templates)}")
    for f in list(templates)[:4]:
        print(f"      {f:<9} {' '.join(x.replace('-CENTRE','C').replace('-WIDE','W') for x in templates[f])}")

    fc = forecast_formations(df)
    v = fc[fc.gameweek >= cut].dropna(subset=["markov"])
    print(f"\n  formation forecast on validation ({len(v)} team-matches):")
    print(f"      Markov chain  {100*(v.markov == v['actual']).mean():.2f}%")
    print(f"      recency       {100*(v.recency == v['actual']).mean():.2f}%")

    # --- ablation grid ---
    rule("ABLATION GRID  (row-level, 5 seeds)")
    configs = {
        "tabular 17 (full)":            TABULAR,
        "tabular + all 3 LLM":          TABULAR + LLM,
        "tabular + rotation only":      TABULAR + ["llm_player_rotation"],
        "tabular + team-rotation only": TABULAR + ["llm_team_rotation"],
        "tabular + sentiment only":     TABULAR + ["llm_manager_sentiment"],
        "recency only (2 feat)":        ["start_rate_last_5", "consecutive_starts"],
        "recency + all 3 LLM":          ["start_rate_last_5", "consecutive_starts"] + LLM,
        "no recency (15 feat)":         [c for c in TABULAR
                                         if c not in ("start_rate_last_5", "consecutive_starts")],
        "no recency + all 3 LLM":       [c for c in TABULAR
                                         if c not in ("start_rate_last_5", "consecutive_starts")] + LLM,
        "LLM only (3 feat)":            LLM,
    }
    print(f"  {'configuration':<30}{'accuracy':>18}{'ROC-AUC':>18}{'F1':>9}")
    print("  " + "-" * 78)
    results, probs = {}, {}
    for name, cols in configs.items():
        p, m = fit_predict(df, cols, cut)
        results[name], probs[name] = m, p
        print(f"  {name:<30}{m['acc']:>10.4f}+/-{m['acc_sd']:.4f}"
              f"{m['auc']:>10.4f}+/-{m['auc_sd']:.4f}{m['f1']:>9.4f}")

    base = results["tabular 17 (full)"]
    rule("LLM DELTA vs ITS OWN TABULAR BASELINE")
    print(f"  {'comparison':<44}{'d acc':>10}{'d auc':>10}{'d f1':>10}")
    print("  " + "-" * 74)
    for a, b in [("tabular 17 (full)", "tabular + all 3 LLM"),
                 ("recency only (2 feat)", "recency + all 3 LLM"),
                 ("no recency (15 feat)", "no recency + all 3 LLM")]:
        x, y = results[a], results[b]
        print(f"  {b + '  vs  ' + a:<44}{y['acc']-x['acc']:>+10.4f}"
              f"{y['auc']-x['auc']:>+10.4f}{y['f1']-x['f1']:>+10.4f}")

    # --- squad level ---
    rule("SQUAD LEVEL  (11 from ~27, three formation conditions)")
    val = df[val_mask].copy()
    for name in ["tabular 17 (full)", "tabular + all 3 LLM"]:
        val["p"] = probs[name]
        print(f"\n  {name}")
        print(f"      {'formation':<12}{'n':>5}{'constrained':>13}{'raw top11':>12}"
              f"{'>=9/11':>9}{'11/11':>8}{'form acc':>10}")
        for mode in ("oracle", "markov", "recency"):
            r = evaluate_squads(val, templates, fc, mode)
            fa = f"{100*r['formation_acc']:.1f}%" if r["formation_acc"] is not None else "  --"
            print(f"      {mode:<12}{r['n']:>5}{100*r['constrained']:>12.2f}%"
                  f"{100*r['raw_top11']:>11.2f}%{100*r['ge9']:>8.1f}%"
                  f"{100*r['perfect']:>7.1f}%{fa:>10}")
        if name.endswith("LLM"):
            r = evaluate_squads(val, templates, fc, "markov")
            print(f"      distribution X/11: "
                  f"{dict(sorted(r['dist'].items(), reverse=True))}")

    # --- comparison with v1 ---
    rule("COMPARISON WITH v1 (paper 1.2)")
    llm = results["tabular + all 3 LLM"]
    val["p"] = probs["tabular + all 3 LLM"]
    sq = evaluate_squads(val, templates, fc, "markov")
    print(f"  {'metric':<34}{'v1 (1.2)':>14}{'v2':>14}{'note':>22}")
    print("  " + "-" * 84)
    print(f"  {'row accuracy':<34}{0.8390:>14.4f}{llm['acc']:>14.4f}{'harder pool':>22}")
    print(f"  {'row ROC-AUC':<34}{0.9074:>14.4f}{llm['auc']:>14.4f}{'':>22}")
    print(f"  {'row F1':<34}{0.8044:>14.4f}{llm['f1']:>14.4f}{'':>22}")
    print(f"  {'constrained lineup acc':<34}{0.7922:>14.4f}{sq['constrained']:>14.4f}"
          f"{'11 of 27, not 20':>22}")
    print(f"  {'lineups >= 9/11':<34}{0.6071:>14.4f}{sq['ge9']:>14.4f}{'':>22}")
    print(f"  {'formation accuracy':<34}{0.8214:>14.4f}"
          f"{(sq['formation_acc'] or 0):>14.4f}{'':>22}")
    d = llm["acc"] - base["acc"]
    print(f"  {'claimed LLM delta (acc)':<34}{0.0715:>14.4f}{d:>14.4f}"
          f"{'controlled':>22}")
    print("\n  v1 numbers come from a different season, an easier candidate pool and")
    print("  labels inferred from start_min; they are context, not a like-for-like test.")


if __name__ == "__main__":
    main()
