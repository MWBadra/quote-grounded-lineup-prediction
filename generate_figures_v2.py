"""Publication figures for the v2 paper. Print-first: one hue for magnitude, a
warm/cool diverging pair only where sign matters, hairline axes, no gridline
clutter, and every value legible at column width in greyscale.

python generate_figures_v2.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

BLUE, RED, GREY = "#2a6fb5", "#c1484a", "#8a8a86"
INK, MUTED, GRID = "#111111", "#5b5b58", "#dcdcd6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.labelcolor": INK,
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def tidy(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


# --- 1. pipeline ---
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    ax.axis("off")
    stages = [
        ("Stage 0/1\nData & availability",
         "FPL-Core-Insights\nconfirmed lineups\n27-player pool"),
        ("Stage 2\nContext extraction",
         "pre-match retrieval\nquote-verified LLM\n2 signals"),
        ("Stage 3\nProbability model",
         "XGBoost\n17 tabular + 12\nposition-relative"),
        ("Stage 4\nAssignment",
         "Hungarian over\n7 coarse slots\n= starting XI"),
    ]
    x = 0.02
    for i, (title, body) in enumerate(stages):
        w = 0.225
        ax.add_patch(FancyBboxPatch((x, 0.18), w, 0.62, boxstyle="round,pad=0.012",
                                    linewidth=1.0, edgecolor=BLUE,
                                    facecolor="#eef4fb", zorder=2))
        ax.text(x + w / 2, 0.68, title, ha="center", va="center",
                fontsize=9.5, weight="bold", zorder=3)
        ax.text(x + w / 2, 0.40, body, ha="center", va="center",
                fontsize=8, color=MUTED, zorder=3)
        if i < 3:
            ax.annotate("", xy=(x + w + 0.017, 0.49), xytext=(x + w + 0.002, 0.49),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))
        x += w + 0.019
    ax.text(0.5, 0.03, "Formation forecasting was evaluated and found to have no "
                       "effect on lineup accuracy (Section IV-D); it is not part of "
                       "the deployed pipeline.",
            ha="center", fontsize=7.6, color=MUTED, style="italic")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    save(fig, "fig1_pipeline")


# --- 2. feature correlations ---
def fig_correlations(df):
    from scipy import stats
    y = df.is_starter.astype(int)
    feats = ["start_rate_last_5", "consecutive_starts", "clean_sheets",
             "recoveries", "clearances_blocks_interceptions", "yellow_cards",
             "f5_accurate_passes_percent", "assists", "goals_scored",
             "f5_chances_created", "f5_total_shots", "availability_score",
             "gk_saves", "llm_player_rotation", "llm_team_rotation"]
    rows = []
    for c in feats:
        if c not in df.columns:
            continue
        m = df[c].notna()
        # n is reported per bar because the correlations are NOT computed on a
        # common sample: the contextual columns exist on a small minority of
        # rows, so an unannotated bar chart invites the reader to compare a
        # coefficient measured on 1,466 rows against one measured on 20,287.
        rows.append((c, stats.pointbiserialr(y[m], df[c][m])[0], int(m.sum())))
    rows.sort(key=lambda t: abs(t[1]))
    # n rides in the tick label rather than beside the bar: the two negative
    # bars are the contextual features, and an inline label on those runs
    # straight into the axis text.
    names = [f"{r[0].replace('_', ' ')}  (n = {r[2]:,})" for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(names, vals, color=[BLUE if v > 0 else RED for v in vals],
            height=0.62, zorder=3)
    lo, hi = min(vals), max(vals)
    pad = 0.22 * max(abs(lo), abs(hi))
    ax.set_xlim(lo - 1.55 * pad, hi + pad)      # extra room for the two negative labels
    for i, v in enumerate(vals):
        off = 0.014 * (hi - lo)
        ax.text(v + (off if v > 0 else -off), i, f"{v:+.3f}",
                va="center", ha="left" if v > 0 else "right",
                fontsize=7.2, color=MUTED, zorder=4)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xlabel("point-biserial correlation with the starting indicator")
    ax.tick_params(axis="y", length=0)
    tidy(ax, "x")
    fig.tight_layout()
    save(fig, "fig2_feature_correlations")


# --- 3. ablation ---
def fig_ablation():
    labels = ["tabular\n(17)", "+ position\npercentiles", "+ LLM\nsignals"]
    # All three configurations scored the same way: the five seeds' probabilities
    # are averaged and the metric computed once on that average, which is also
    # the vector Stage 4 consumes. Mixing that with a mean-of-per-seed-metrics
    # figure (0.8365 for the tabular row) overstated the percentile increment.
    acc = [0.8372, 0.8454, 0.8462]
    auc = [0.9168, 0.9180, 0.9220]
    ge9 = [63.6, 67.8, 70.3]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.7))
    for ax, vals, title, fmt, lo in [
            (axes[0], acc, "Row accuracy", "{:.4f}", 0.83),
            (axes[1], auc, "Row ROC-AUC", "{:.4f}", 0.91),
            (axes[2], ge9, r"Lineups $\geq$9/11 correct (%)", "{:.1f}", 60)]:
        shades = ["#b8cfe8", "#6b9fd0", BLUE]
        ax.bar(labels, vals, color=shades, width=0.6, zorder=3)
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom",
                    fontsize=8, weight="bold")
        ax.set_title(title, pad=8)
        ax.set_ylim(lo, max(vals) * 1.012 if lo < 1 else max(vals) * 1.06)
        tidy(ax)
    fig.tight_layout()
    save(fig, "fig3_ablation")


# --- 4. cross-validation ---
def fig_cv():
    # Five seeds, missing values left missing -- the same fitting protocol as
    # Table I, so the folds are comparable with the headline configuration.
    folds = ["18-21", "22-25", "26-29", "30-33", "34-38"]
    acc = [0.8613, 0.8468, 0.8540, 0.8513, 0.8314]
    con = [81.82, 81.82, 82.15, 82.15, 77.61]
    ge9 = [68.4, 69.3, 74.1, 75.6, 60.0]
    fig, ax1 = plt.subplots(figsize=(5.6, 3.1))
    ax1.plot(folds, con, "-o", color=BLUE, lw=1.8, ms=6, label="constrained lineup acc.",
             zorder=3)
    ax1.plot(folds, ge9, "-s", color=GREY, lw=1.6, ms=5.5,
             label=r"lineups $\geq$9/11", zorder=3)
    ax1.axhline(np.mean(con), color=BLUE, lw=0.8, ls=(0, (4, 3)), alpha=0.6)
    ax1.set_ylabel("per cent")
    ax1.set_xlabel("test gameweeks (forward chaining)")
    ax1.set_ylim(50, 90)
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.annotate("end-of-season\nrotation", xy=(4, 60.0), xytext=(3.05, 53.5),
                 fontsize=7.6, color=RED, ha="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    tidy(ax1)
    save(fig, "fig4_cv_stability")


# --- 5. lineup distribution ---
def fig_distribution():
    dist = {11: 16, 10: 67, 9: 83, 8: 43, 7: 16, 6: 5, 5: 1, 4: 2, 3: 2, 2: 1}
    ks = sorted(dist, reverse=True)
    vals = [100 * dist[k] / sum(dist.values()) for k in ks]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    cols = [BLUE if k >= 9 else "#c3d4e6" for k in ks]
    ax.bar([str(k) for k in ks], vals, color=cols, width=0.68, zorder=3)
    for i, (k, v) in enumerate(zip(ks, vals)):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.6, color=MUTED)
    ax.set_xlabel("correctly predicted starters out of 11")
    ax.set_ylabel("% of validation squads")
    ax.set_ylim(0, max(vals) * 1.34)
    ax.text(0.98, 0.95, "70.3% of squads reach 9/11 or better",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.4, color=BLUE, weight="bold")
    tidy(ax)
    save(fig, "fig5_lineup_distribution")


# --- 6. leakage ablation ---
def fig_leakage():
    arms = ["A  sourced\n(quotes verified)", "B  memory only\n(no sources)",
            "C  shuffled\nsources"]
    auc = [0.711, 0.654, 0.598]
    cov = [10.7, 68.4, 1.5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 2.9))
    a1.bar(arms, auc, color=[BLUE, RED, GREY], width=0.6, zorder=3)
    a1.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    a1.text(-0.42, 0.505, "chance", fontsize=7.2, color=MUTED, ha="left",
            va="bottom")
    for i, v in enumerate(auc):
        a1.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.4, weight="bold")
    a1.set_ylim(0.45, 0.77)
    a1.set_ylabel("ROC-AUC of the extracted signal")
    a1.set_title("Discrimination", pad=8)
    tidy(a1)

    a2.bar(arms, cov, color=[BLUE, RED, GREY], width=0.6, zorder=3)
    for i, v in enumerate(cov):
        a2.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.4, weight="bold")
    a2.set_ylabel("% of player-matches carrying a signal")
    a2.set_title("Coverage", pad=8)
    a2.set_ylim(0, 78)
    tidy(a2)
    fig.tight_layout()
    save(fig, "fig6_leakage_ablation")


# --- 7. formation conditions ---
def fig_formation():
    # Measured on the proposed configuration (tabular + percentiles + LLM), not
    # on the weaker tabular+LLM model the first version of this analysis used.
    # The fourth bar is the control that matters: a single formation imposed on
    # every fixture, ignoring the shape entirely.
    conds = ["true formation\n(oracle)", "Markov chain\n(69.9% correct)",
             "recency\n(70.3% correct)", "constant 4-2-3-1\n(ignores the fixture)"]
    con = [81.01, 80.89, 80.93, 80.89]
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.bar(conds, con, color=["#b8cfe8", BLUE, "#6b9fd0", GREY], width=0.58, zorder=3)
    for i, v in enumerate(con):
        ax.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontsize=9, weight="bold")
    ax.set_ylim(80.0, 81.35)
    ax.set_ylabel("constrained lineup accuracy")
    ax.set_title("Knowing the true formation confers no practical advantage",
                 pad=8, fontsize=9.5)
    tidy(ax)
    save(fig, "fig7_formation_conditions")


# --- 8. baselines ---
def fig_baselines():
    names = ["random", "availability\nonly", "start rate\nonly",
             "consecutive\nstarts only", "full\npipeline"]
    # Full-pipeline row is the SAME model as Table I and Section V-F (five-seed
    # ensemble, native missing-value handling), so the three places the paper
    # quotes a constrained lineup accuracy now quote one number.
    con = [41.41, 48.04, 75.81, 76.31, 80.89]
    ge9 = [0.4, 0.8, 52.1, 56.4, 70.3]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x - 0.19, con, 0.38, label="constrained lineup acc.",
           color=BLUE, zorder=3)
    ax.bar(x + 0.19, ge9, 0.38, label=r"lineups $\geq$9/11", color="#c3d4e6",
           zorder=3)
    for i, (a, b) in enumerate(zip(con, ge9)):
        ax.text(i - 0.19, a + 1, f"{a:.1f}", ha="center", fontsize=7.2, color=MUTED)
        ax.text(i + 0.19, b + 1, f"{b:.1f}", ha="center", fontsize=7.2, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("per cent")
    ax.set_ylim(0, 104)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(0.0, 1.02))
    tidy(ax)
    save(fig, "fig8_baselines")


def main():
    print("writing figures to version 2/figures/")
    df = pd.read_csv(HERE / "dataset" / "pl_2025_2026_final.csv", encoding="utf-8-sig")
    fig_pipeline()
    fig_correlations(df)
    fig_ablation()
    fig_cv()
    fig_distribution()
    fig_leakage()
    fig_formation()
    fig_baselines()


if __name__ == "__main__":
    main()
