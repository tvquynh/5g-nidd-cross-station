# What Cross-Station Evaluation Hides

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23190918.svg)](https://doi.org/10.5281/zenodo.23190918)


Reproducibility artifact for the manuscript *What Cross-Station Evaluation
Hides: Direction Asymmetry and Scoring-Set Composition in 5G Intrusion
Detection*, under review at IEEE Transactions on Network and Service
Management.

Every number in the article resolves to a file here. The pipeline, the
per-cell result records, the aggregation scripts that generate each table and
figure, and the tests all ship. The 5G-NIDD dataset does not: its authors
distribute it, and this repository consumes their published release rather
than redistributing it.

## What the article measures

Cross-station evaluation, training an intrusion detector on one base station
and scoring it on another, is the standard test of whether a 5G detector
generalizes beyond its capture environment. The article shows that the protocol
as commonly applied misreports what it measures, and quantifies by how much.

1. **Degradation is direction-asymmetric.** Detectors trained on the
   attack-heavy station flag 69 to 70 percent of benign flows from the other
   station as attacks; the reverse direction yields 0.2 to 1.7 percent. The
   pattern holds across seven model families, including two deep tabular
   architectures, so it is a property of the data rather than of any learner. A
   single bidirectional average, which is the customary report, describes
   neither direction.

2. **The reported magnitude belongs to the scoring set.** Aggregate binary F1
   for a fixed detector moves by 0.231 when only the scoring set's class mix
   changes, while its false-positive rate moves by at most 0.0024. A detector
   with a 69 percent false-positive rate therefore reports an F1 of 0.92, and
   ranking protocols by F1 places the operationally unusable configuration
   above the merely imperfect one. A same-station prior control confirms that
   class priors are not the cause.

## The 652 runs, and how to check the count

```bash
python scripts/count_runs.py
```

| set | runs | what it is |
|---|---|---|
| `results/grid/` | 200 | five lightweight families, 300k subsample |
| `results/grid_deep/` | 40 | two deep tabular families, 300k subsample |
| `results/grid_full_1m/` | 12 | full 1.21M verification |
| `results/decomposition/` | 400 | composition design and same-station prior control |
| | **652** | |

The four sets are separate directories because a run is identified by its
protocol, transfer direction, model family, seed, **and the size of the
training set it saw**. The cross-station runs were repeated on all 1.21M flows
as a check on the subsample, and in our working tree those repeats were stored
under two names. Counting files there would have double-counted them. Here each
run appears exactly once, and the script fails loudly if the total moves.

## Getting it running

```bash
python -m venv .venv && . .venv/bin/activate      # or .venv\Scripts\activate
pip install -r requirements.txt
```

Obtain 5G-NIDD from its authors and place their `Encoded.csv` at
`data/raw/combined_encoded/Encoded.csv`. The loader asserts the file's row
count and per-class counts on load, so a divergence in the input is detected
rather than silently absorbed. Paths live in `configs/paths.yaml`.

Reproduce the tables and figures from the shipped records, without retraining:

```bash
python -m src.aggregate_results
python -m src.aggregate_decomposition
python -m src.make_paper_tables
python -m src.make_paper_figures
```

Rerun the campaigns from scratch, which is CPU-only and takes days, dominated
by FT-Transformer:

```bash
python scripts/run_campaign.py --stage grid
python scripts/run_campaign.py --stage decomposition
```

Run the tests:

```bash
python -m pytest -q
```

`tests/test_splits_shift_decomposition.py` covers the protocol definitions:
whether a split leaks across stations, whether class matching equalizes the
sides it claims to, and whether the composition cells are what they say.
`tests/test_stats.py` covers the statistical treatment, including the floor on
the two-sided p-value at ten paired seeds, which is why the article reports
`p < 0.002` rather than a smaller figure.

## Protocols

Four, and the cross-station one is always reported per direction:

- **Random**, stratified 80/20 over the multiclass label. An in-distribution
  upper bound, used as a reference rather than as a result.
- **Within-day temporal**, the last 30 percent of each (station, attack type)
  group by capture order. Isolates drift from attack novelty.
- **Cross-station**, fit on one station and score on the other, both
  directions, never averaged. In the class-matched form, per-class counts are
  equalized on both sides by drawing the smaller of the two counts, so no
  synthetic upsampling is introduced.
- **Prior control**, both sides drawn from the same station with only the class
  priors altered. Separates a prior effect from a station effect.

## Layout

```
configs/       paths and model hyperparameters
src/           loader, preprocessing, splits, models, training, aggregation, figures
scripts/       run_campaign.py (campaign driver), count_runs.py (record audit)
tests/         protocol definitions and statistical treatment
results/       per-cell records, aggregated tables, campaign logs
```

`src/` carries the model factory for a stable-feature-subset variant that this
article does not report. It is present because `src/models.py` imports it and
the module set has to close under its own imports.

## Scope

A companion manuscript by an overlapping author group studies
out-of-distribution scoring rules on held-out attack types and is archived
separately. The two studies share this data loader, this job driver and the
same fixed seed list, which is what reproducibility requires of companion work
from one group. They share no experiment: nothing here holds out an attack
type, and there is no out-of-distribution or open-set content in this
repository.

## Citing

See `CITATION.cff`. Cite both the article and this artifact.

## License

MIT, see `LICENSE`. The 5G-NIDD dataset carries its authors' own terms.
