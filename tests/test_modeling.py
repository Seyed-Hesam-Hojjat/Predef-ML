import numpy as np
import pandas as pd

from ml_workbench.modeling import benchmark_models, evaluate_final_model, make_holdout_split
from ml_workbench.preprocessing import prepare_features


def _synthetic_df(n=100):
    rng = np.random.default_rng(123)
    x = rng.normal(size=n)
    category = rng.choice(["a", "b", "c"], size=n)
    logits = x + (category == "a") * 0.7 + rng.normal(size=n) * 0.7
    y = (logits > 0.1).astype(int)
    return pd.DataFrame({"x": x, "category": category, "outcome": y})


def test_group_split_keeps_groups_separate():
    df = _synthetic_df(120)
    X, y, _ = prepare_features(df, "outcome", excluded_columns=["group"] if "group" in df else [])
    groups = pd.Series(np.repeat(np.arange(30), 4))
    train_idx, test_idx = make_holdout_split(
        X, y, "classification", strategy="Group-based (same group never crosses splits)",
        groups=groups, test_size=0.25,
    )
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))


def test_time_split_uses_later_rows_as_holdout():
    n = 100
    times = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    X = pd.DataFrame({"x": np.arange(n)})
    y = pd.Series(np.arange(n, dtype=float))
    train_idx, test_idx = make_holdout_split(
        X, y, "regression", strategy="Time-based (train on earlier, test on later rows)",
        times=times, test_size=0.2,
    )
    assert times.iloc[train_idx].max() < times.iloc[test_idx].min()


def test_end_to_end_benchmark_and_final_evaluation():
    df = _synthetic_df(110)
    X, y, _ = prepare_features(df, "outcome")
    train_idx, test_idx = make_holdout_split(X, y, "classification", test_size=0.2)
    benchmark = benchmark_models(
        X, y, "classification", train_idx, test_idx,
        "Random (stratified for classification)", primary_metric="roc_auc",
        requested_cv_splits=3, feature_selection="Model-based (Extra Trees)", k_features=3,
    )
    assert benchmark["best_model_name"]
    assert (benchmark["cv_table"]["status"] == "ok").any()
    final = evaluate_final_model(
        benchmark, X, y, "classification", permutation_repeats=2,
        row_ids=list(range(2, len(df) + 2)), class_label_mapping={0: "no", 1: "yes"},
    )
    assert "accuracy" in final["metrics"]
    assert len(final["predictions"]) == len(test_idx)
    assert set(final["predictions"]["actual"]).issubset({"no", "yes"})
    assert "source_row_number" in final["predictions"]
    assert not final["permutation_importance"].empty


def test_end_to_end_regression_metrics():
    rng = np.random.default_rng(2026)
    n = 90
    x = rng.normal(size=n)
    category = rng.choice(["low", "medium", "high"], size=n)
    y = 12 + 3.5 * x + (category == "high") * 2 + rng.normal(size=n) * 0.8
    df = pd.DataFrame({"x": x, "category": category, "outcome": y})
    X, target, _ = prepare_features(df, "outcome")
    train_idx, test_idx = make_holdout_split(X, target, "regression", test_size=0.2)
    benchmark = benchmark_models(
        X, target, "regression", train_idx, test_idx,
        "Random (stratified for classification)", primary_metric="rmse", requested_cv_splits=3,
    )
    final = evaluate_final_model(benchmark, X, target, "regression", permutation_repeats=2)
    assert final["metrics"]["rmse"] >= 0
    assert final["metrics"]["mae"] >= 0
    assert len(final["predictions"]) == len(test_idx)
