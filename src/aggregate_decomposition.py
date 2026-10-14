"""Aggregate the factorial decomposition of cross-station degradation.

The cross-station literature reports a single number: train on one base
station, score on the other. That number moves for two separable reasons ---
the class composition the detector is trained on, and the class composition it
is scored against --- and a protocol that rebalances both at once cannot say
which. This script assembles the two-by-two design into per-model tables and
tests each main effect with a paired Wilcoxon signed-rank test over the seeds.

It also contrasts prevalence-sensitive metrics (binary F1) with
prevalence-independent ones (true-positive rate, false-positive rate). Where a
large difference in binary F1 is accompanied by an unchanged TPR and FPR, the
degradation is a property of the scoring set's composition rather than of the
detector's ability to discriminate.

Usage:
    python -m src.aggregate_decomposition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_experiment import load_paths

# The four cells of the design plus the same-station control. The fully
# matched cell is produced by the grid stage under its original name.
CELL_ALIASES = {
    "cross_station_raw_raw": ("raw", "raw"),
    "cross_station_raw_matched": ("raw", "matched"),
    "cross_station_matched_raw": ("matched", "raw"),
    "cross_station_matched_matched": ("matched", "matched"),
    "cross_station": ("matched", "matched"),
    "cross_station_naive": ("raw", "raw"),
}

METRICS = ["macro_f1", "binary_f1", "binary_recall", "binary_fpr"]
METRIC_LABELS = {
    "macro_f1": "macro-F1",
    "binary_f1": "binary F1",
    "binary_recall": "binary TPR",
    "binary_fpr": "binary FPR",
}


# The subsample campaign draws 300,000 flows; a run above this is full-scale.
SUBSAMPLE_MAX_TRAIN = 300_000


def load_records(directories: List[Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for directory in directories:
        if not directory.exists():
            continue
        for file in sorted(directory.glob("*.json")):
            record = json.loads(file.read_text())
            split = record.get("split")
            if split not in CELL_ALIASES and split != "prior_control":
                continue
            if record.get("features") != "full":
                continue
            # The full-scale verification runs share a cell identity with the
            # subsample runs they check, so without this they collide. The
            # tables report the subsample campaign; the full-scale runs are
            # cited separately in the experimental setup.
            if (record.get("n_train") or 0) > SUBSAMPLE_MAX_TRAIN:
                continue
            row = {
                "model": record["model"],
                "protocol": split,
                "train_bs": record.get("train_bs") or 0,
                "seed": record["seed"],
                "n_train": record.get("n_train"),
                "n_test": record.get("n_test"),
            }
            if split == "prior_control":
                row["train_balance"], row["test_balance"] = "same-station", "other priors"
            else:
                row["train_balance"], row["test_balance"] = CELL_ALIASES[split]
            for metric in METRICS:
                row[metric] = record.get(metric)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # A configuration can appear in more than one source directory. Collapsing
    # that silently is how two tables in one paper came to report different
    # values for the same cell: whichever directory was listed first won, and
    # nothing said so. Identical rows collapse; rows that disagree stop the run.
    key = ["model", "protocol", "train_bs", "seed"]
    conflicts = []
    for identity, group in frame.groupby(key, dropna=False):
        if len(group) == 1:
            continue
        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric].dropna().unique()
            if len(values) > 1:
                conflicts.append((identity, metric, sorted(values)))
    if conflicts:
        lines = [f"{len(conflicts)} cell/metric pairs disagree between source "
                 "directories. Pick one source rather than letting directory "
                 "order decide:"]
        for identity, metric, values in conflicts[:10]:
            shown = ", ".join(f"{v:.9f}" for v in values)
            lines.append(f"  {identity} {metric}: {shown}")
        if len(conflicts) > 10:
            lines.append(f"  ... and {len(conflicts) - 10} more")
        raise SystemExit("\n".join(lines))

    return frame.drop_duplicates(subset=key)


def paired_effect(frame: pd.DataFrame, metric: str,
                  protocol_a: str, protocol_b: str) -> Optional[Dict]:
    """Paired difference protocol_b minus protocol_a, tested over seeds."""
    a = frame[frame["protocol"] == protocol_a].set_index("seed")[metric]
    b = frame[frame["protocol"] == protocol_b].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 6:
        return None
    delta = (b.loc[common] - a.loc[common]).values
    if np.allclose(delta, 0):
        return {"n": len(common), "mean": 0.0, "std": 0.0, "p_value": 1.0, "cohens_d": 0.0}
    statistic, p_value = wilcoxon(delta, alternative="two-sided")
    spread = float(delta.std(ddof=1))
    return {
        "n": int(len(common)),
        "mean": float(delta.mean()),
        "std": spread,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "cohens_d": float(delta.mean() / spread) if spread > 0 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="binary_f1",
                    help="metric used for the printed effect tables")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    paths = load_paths()
    results = Path(paths["results"])
    # results/metrics/ is canonical for the accuracy grid. results/metrics_windows/
    # is the same experiments in a second environment and is NOT a source here;
    # it is compared against this one by src/compare_environments.py. Adding it
    # back would reintroduce two different values for the same cell.
    frame = load_records([
        results / "decomposition",
        results / "metrics",
    ])
    if frame.empty:
        raise SystemExit("no decomposition records found")

    print(f"Loaded {len(frame)} runs "
          f"({frame['model'].nunique()} models, {frame['protocol'].nunique()} protocols)")

    summary = frame.groupby(["model", "train_bs", "protocol", "train_balance", "test_balance"]).agg(
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, "std") for m in METRICS},
        n_seeds=("seed", "nunique"),
    ).reset_index()

    for train_bs in sorted(frame["train_bs"].unique()):
        other = 2 if train_bs == 1 else 1
        block = summary[summary["train_bs"] == train_bs]
        if block.empty:
            continue
        print(f"\n{'=' * 78}")
        print(f"Direction BS{train_bs} -> BS{other}")
        print("=" * 78)
        columns = ["model", "train_balance", "test_balance", "n_seeds",
                   "macro_f1_mean", "binary_f1_mean", "binary_recall_mean", "binary_fpr_mean"]
        with pd.option_context("display.width", 220):
            print(block.sort_values(["model", "train_balance", "test_balance"])[columns]
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Main effects, tested per model and direction.
    effect_rows: List[Dict] = []
    effects = [
        ("test composition (train raw)", "cross_station_raw_raw", "cross_station_raw_matched"),
        ("test composition (train matched)", "cross_station_matched_raw", "cross_station"),
        ("train composition (test raw)", "cross_station_raw_raw", "cross_station_matched_raw"),
        ("train composition (test matched)", "cross_station_raw_matched", "cross_station"),
    ]
    for metric in METRICS:
        for (model, train_bs), group in frame.groupby(["model", "train_bs"]):
            for label, protocol_a, protocol_b in effects:
                effect = paired_effect(group, metric, protocol_a, protocol_b)
                if effect is None:
                    continue
                effect_rows.append({
                    "metric": metric, "model": model, "train_bs": train_bs,
                    "effect": label, **effect,
                })
    effects_frame = pd.DataFrame(effect_rows)

    if not effects_frame.empty:
        metric = args.metric
        print(f"\n{'=' * 78}")
        print(f"Main effects on {METRIC_LABELS.get(metric, metric)} "
              f"(paired over seeds, Wilcoxon)")
        print("=" * 78)
        block = effects_frame[effects_frame["metric"] == metric]
        with pd.option_context("display.width", 220):
            print(block[["model", "train_bs", "effect", "n", "mean", "std", "p_value", "cohens_d"]]
                  .sort_values(["train_bs", "model", "effect"])
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        # The headline contrast: how large the test-composition effect is on a
        # prevalence-sensitive metric versus on prevalence-independent ones.
        print(f"\n{'=' * 78}")
        print("Test-composition effect by metric (mean over models and directions)")
        print("=" * 78)
        test_effects = effects_frame[effects_frame["effect"].str.startswith("test composition")]
        pivot = test_effects.groupby("metric").agg(
            mean_effect=("mean", "mean"),
            max_abs_effect=("mean", lambda s: float(np.abs(s).max())),
            n_tests=("mean", "size"),
            n_significant=("p_value", lambda s: int((s < 0.05).sum())),
        ).reset_index()
        pivot["metric"] = pivot["metric"].map(METRIC_LABELS).fillna(pivot["metric"])
        print(pivot.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    prefix = Path(args.out_prefix) if args.out_prefix else results / "decomposition_summary"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(f"{prefix}_per_seed.csv", index=False)
    summary.to_csv(f"{prefix}_cells.csv", index=False)
    if not effects_frame.empty:
        effects_frame.to_csv(f"{prefix}_effects.csv", index=False)
    print(f"\nSaved: {prefix}_cells.csv, {prefix}_effects.csv")


if __name__ == "__main__":
    main()
