"""Tests for the statistical treatment the manuscript's claims rest on.

Every significance statement in the paper comes out of src/stats.py, so these
check the three things that would silently change a reported conclusion: the
effect size, the paired test, and the floor on the p-value at ten paired
samples.
"""
import numpy as np
import pandas as pd
import pytest

from src.stats import ci95, cohens_d, compare, summary_table


def test_cohens_d_sign_follows_argument_order():
    """A positive d means the first argument is larger. The paper reads the
    sign, so a flip here would invert a reported direction."""
    high = [0.9, 0.91, 0.89, 0.90, 0.92]
    low = [0.5, 0.51, 0.49, 0.50, 0.52]
    assert cohens_d(high, low) > 0
    assert cohens_d(low, high) < 0
    assert cohens_d(high, low) == pytest.approx(-cohens_d(low, high))


def test_cohens_d_is_zero_without_a_difference():
    x = [0.7, 0.8, 0.9, 0.6, 0.75]
    assert cohens_d(x, x) == pytest.approx(0.0)


def test_cohens_d_survives_zero_variance():
    """Repeated runs inside one environment are bit-identical in this pipeline,
    so a zero-variance pair reaches this function and must not divide by zero."""
    assert cohens_d([0.5] * 5, [0.5] * 5) == 0.0


def test_ci95_brackets_the_mean_and_widens_with_spread():
    tight = [0.900, 0.901, 0.899, 0.900, 0.901]
    loose = [0.80, 0.95, 0.85, 0.99, 0.90]
    m_t, lo_t, hi_t = ci95(tight)
    m_l, lo_l, hi_l = ci95(loose)
    assert lo_t <= m_t <= hi_t
    assert lo_l <= m_l <= hi_l
    assert (hi_l - lo_l) > (hi_t - lo_t)


def test_ci95_degenerates_safely_on_one_sample():
    m, lo, hi = ci95([0.42])
    assert m == pytest.approx(0.42)
    assert lo == 0.0 and hi == 0.0


def _frame(values_by_split, model="lightgbm", features="full"):
    rows = []
    for split, values in values_by_split.items():
        for seed, value in enumerate(values):
            rows.append({"model": model, "features": features, "split": split,
                         "seed": seed, "macro_f1": value})
    return pd.DataFrame(rows)


def test_wilcoxon_floor_at_ten_paired_seeds():
    """With ten paired seeds the smallest attainable two-sided p-value is
    0.001953. The paper reports p < 0.002 and reads it as the strongest the
    test can express at this sample size, not as an unusually strong effect.
    A change in scipy's default that moved this floor would change how those
    claims should be worded."""
    a = np.linspace(0.90, 0.99, 10)
    b = a - 0.10                      # every pair differs in the same direction
    df = _frame({"random": a, "cross_station": b})
    out = compare(df)
    assert len(out) == 1
    row = out[0]
    assert row["n_seeds"] == 10
    assert row["wilcoxon_p"] == pytest.approx(0.001953, abs=1e-6)
    assert row["wilcoxon_p"] >= 0.001953


def test_compare_pairs_only_within_model_and_feature_set():
    """Pooling across model families would average away the direction
    asymmetry, which is the paper's first result."""
    a = np.linspace(0.90, 0.99, 10)
    df = pd.concat([
        _frame({"random": a, "cross_station": a - 0.1}, model="lightgbm"),
        _frame({"random": a, "cross_station": a - 0.2}, model="rf"),
    ], ignore_index=True)
    out = compare(df)
    models = sorted(row["model"] for row in out)
    assert models == ["lightgbm", "rf"]
    # Splits are paired in alphabetical order, so split_a is "cross_station"
    # and diff is cross_station minus random, i.e. negative here. Reading the
    # sign without checking which split is a would invert the direction.
    by_model = {row["model"]: row["diff"] for row in out}
    assert all(row["split_a"] == "cross_station" for row in out)
    assert by_model["lightgbm"] == pytest.approx(-0.1, abs=1e-9)
    assert by_model["rf"] == pytest.approx(-0.2, abs=1e-9)


def test_compare_skips_pairs_with_too_few_seeds():
    """Two seeds cannot support a paired test and must be dropped rather than
    reported with a meaningless p-value."""
    df = _frame({"random": [0.9, 0.91], "cross_station": [0.8, 0.81]})
    assert compare(df) == []


def test_compare_reports_the_difference_in_the_stated_order():
    a = np.linspace(0.90, 0.99, 10)
    df = _frame({"random": a, "cross_station": a - 0.1})
    row = compare(df)[0]
    assert row["split_a"] < row["split_b"]
    expected = row["mean_a"] - row["mean_b"]
    assert row["diff"] == pytest.approx(expected)


def test_summary_table_counts_seeds_per_cell():
    """The deep tabular families run five seeds and the lightweight ones ten.
    The tables state which applies, so the count has to come from the data."""
    df = pd.concat([
        _frame({"random": np.linspace(0.9, 0.99, 10)}, model="lightgbm"),
        _frame({"random": np.linspace(0.9, 0.95, 5)}, model="ftt"),
    ], ignore_index=True)
    rows = {row["model"]: row["n_seeds"] for row in summary_table(df)}
    assert rows == {"lightgbm": 10, "ftt": 5}
