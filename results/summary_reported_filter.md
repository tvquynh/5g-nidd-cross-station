# What was filtered out of the evidence files, and why

Two aggregate files in this kit are restricted to what the manuscript
reports. Nothing was removed because it was inconvenient; everything
removed belongs to a section that was cut when this study moved from a
22-page Elsevier format to the 10-page IEEE TNSM format, or to the
companion open-set study.

## summary_reported.csv

Source: `results/summary.csv`, 1198 rows.
Kept: **240** rows, the subsample accuracy grid, 200 lightweight cells and 40 deep tabular ones.
Removed: **958** rows.

| removed | rows |
|---|---|
| protocol not reported here: holdout_attack | 212 |
| feature budget not reported here: top20 | 200 |
| feature budget not reported here: top20_stable | 200 |
| feature budget not reported here: top50 | 200 |
| record from the companion open-set study | 134 |
| model variant not reported here: stability_lgbm | 11 |
| smoke run | 1 |

## stats_reported.json

Source: `results/stats.json`, 92 summary rows and 135 pairwise comparisons.
Kept: **21** summary rows and **21** comparisons.

| removed | entries |
|---|---|
| feature budget not reported here: top20 | 50 |
| feature budget not reported here: top20_stable | 50 |
| feature budget not reported here: top50 | 50 |
| protocol not reported here: holdout_attack | 29 |
| model variant not reported here: stability_lgbm | 6 |

## decomposition_summary_*_reported.csv

| file | rows in | rows kept |
|---|---|---|
| `decomposition_summary_cells.csv` | 56 | 54 |
| `decomposition_summary_effects.csv` | 160 | 160 |
| `decomposition_summary_per_seed.csv` | 525 | 520 |

| removed | rows |
|---|---|
| model variant not reported here: stability_lgbm | 7 |

The deep tabular rows are kept here: the architecture check is one of the
things this manuscript reports.

## Where the unfiltered data is

The 240 rows above are the subsample accuracy grid. They are not the
manuscript's run total of 652, which also counts 12 full-scale
verification runs and 400 cells of the composition design.
`scripts/count_runs.py` in the artifact is the authoritative count.

Every per-cell record, filtered or not, is public at
`https://github.com/tvquynh/5g-nidd-cross-station`, DOI
`10.5281/zenodo.23190918`. The unfiltered aggregates stay in the working
tree.
