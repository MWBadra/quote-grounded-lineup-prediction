# Quote-Grounded Language Model Context for Football Starting Lineup Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21903339.svg)](https://doi.org/10.5281/zenodo.21903339)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
Code and data for the paper *Quote-Grounded Language Model Context for Football
Starting Lineup Prediction: Measuring Rather Than Assuming Contamination*.

Predicts which eleven players a Premier League manager will start, before kickoff,
by combining tabular match data with contextual signals pulled out of pre-match
reporting by an LLM. Every extracted signal has to cite a verbatim span that is
then verified against the retrieved text in code, so anything the model recalled
rather than read gets thrown away.

Evaluated on the whole 2025-26 season: 370 fixtures, 740 team-matches, 20,287
player-match rows, with confirmed team sheets as labels.

## Headline numbers

| | |
|---|---|
| Nine or more of eleven starters correct | 70.3% of 236 validation squads |
| Same, best no-model recency rule | 56.4% |
| Constrained lineup accuracy | 80.89% |
| Row-level ROC-AUC | 0.9220 |

The contextual block is worth +0.0040 ROC-AUC pooled, and +0.0363 on the 7.23%
of rows it actually reaches. The paper argues at some length about why the small
pooled number is the expected result rather than a disappointing one.

## Layout

```
build_dataset.py            stage 0/1  tabular dataset + availability pool
explore_dataset.py                     inventories the upstream repo first
retrieval.py                stage 2    pre-match corpus, Tavily
extraction.py               stage 2    quote-grounded LLM extraction
llm_keys.py                            key pool, quota, rate limiting
run_full.py                            full-season extraction, resumable
run_pipeline_v2.py          stage 3/4  probabilities, formation, assignment
positional_percentiles.py              position-relative percentile block
cross_validation.py                    forward-chaining folds + baselines
leakage_ablation.py                    the three-arm contamination ablation
llm_significance.py                    McNemar, bootstrap CIs, encodings
formation_value.py                     what knowing the formation is worth
monotone_ablation.py                   cost of monotone constraints
error_analysis.py                      where the tabular model fails
generate_figures_v2.py                 figures
paper.tex                              the paper
```

`dataset/` has the built tables, including `llm_signals_clean.csv` (the extracted
signals with their quotes and source types). `reports/` has the correlation and
feature-selection output the methods section is written from.

## Running it

```
pip install pandas numpy scipy scikit-learn xgboost matplotlib
```

The caches aren't in the repo, so the first run rebuilds them:

```
python explore_dataset.py          # inventory the upstream repo
python build_dataset.py            # tabular dataset
python run_full.py --stage all     # retrieval + extraction, costs credits
python run_pipeline_v2.py          # evaluation
```

Retrieval and extraction need `TAVILY_API_KEY` and one or more `GEMINI_API_KEY*`
in a `.env` next to the scripts. Both stages cache per (team, gameweek) and the
cache is the checkpoint, so a re-run picks up wherever it stopped. Run
`python dry_run.py` first if you want to exercise the filters and the quote
verifier without spending anything, and `python pilot.py` for a five-fixture
live test before committing the full budget.

Everything downstream of `run_pipeline_v2.py` runs off the built dataset and
needs no network access or API keys.

## What isn't here, and why

The retrieval caches hold the full text of articles from Premier League and
other publishers. That isn't mine to redistribute, so `.newscache*/` and
`.extractcache/` are gitignored. `.cache/` is a local mirror of the upstream
[FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights) CSVs and is
left out for the same reason. The extracted signals themselves are in
`dataset/`, including the quote each one is grounded in, so the extraction is
auditable without the corpus.

## Data

Match and player data from
[FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights), itself
derived from the official Fantasy Premier League API. Pre-match text retrieved
via Tavily, date-bounded to close the day before kickoff.
