"""Model benchmarking with training-only cross-validation and a final holdout test."""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, KFold, RandomizedSearchCV, StratifiedKFold,
    TimeSeriesSplit, cross_validate, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .preprocessing import estimate_dense_cells, make_preprocessor

RANDOM_STATE = 42
DENSE_CELL_LIMIT = 5_000_000  # ~40 MB just for one float64 matrix, before model copies


class SafeSelectKBest(SelectKBest):
    """SelectKBest variant that caps k at the transformed fold's feature count."""

    def fit(self, X: Any, y: Any = None) -> "SafeSelectKBest":
        if isinstance(self.k, (int, np.integer)) and int(self.k) > int(X.shape[1]):
            self.k = int(X.shape[1])
        return super().fit(X, y)


def metric_catalog(task: str, n_classes: int = 2) -> dict[str, dict[str, Any]]:
    """Scoring map. sklearn's negative-loss convention is converted for display later."""
    if task == "classification":
        result: dict[str, dict[str, Any]] = {
            "accuracy": {"scoring": "accuracy", "direction": "max", "label": "Accuracy"},
            "balanced_accuracy": {"scoring": "balanced_accuracy", "direction": "max", "label": "Balanced accuracy"},
            "f1_macro": {"scoring": "f1_macro", "direction": "max", "label": "Macro F1"},
        }
        if n_classes == 2:
            result["roc_auc"] = {"scoring": "roc_auc", "direction": "max", "label": "ROC AUC"}
            result["average_precision"] = {"scoring": "average_precision", "direction": "max", "label": "Average precision"}
        else:
            result["roc_auc_ovr"] = {"scoring": "roc_auc_ovr", "direction": "max", "label": "ROC AUC (one-vs-rest)"}
        return result
    return {
        "rmse": {"scoring": "neg_root_mean_squared_error", "direction": "min", "label": "RMSE"},
        "mae": {"scoring": "neg_mean_absolute_error", "direction": "min", "label": "MAE"},
        "r2": {"scoring": "r2", "direction": "max", "label": "R²"},
    }


def _model_specs(task: str, random_state: int, allow_dense: bool) -> list[dict[str, Any]]:
    """Return candidate algorithms plus compact randomized-search spaces."""
    if task == "classification":
        specs = [
            {"name": "Dummy baseline", "estimator": DummyClassifier(strategy="prior"), "dense": False, "params": {}},
            {"name": "Logistic regression", "estimator": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state), "dense": False,
             "params": {"model__C": loguniform(1e-3, 1e2), "model__class_weight": [None, "balanced"]}},
            {"name": "Ridge classifier", "estimator": RidgeClassifier(class_weight="balanced"), "dense": False,
             "params": {"model__alpha": [0.1, 1.0, 10.0, 100.0], "model__class_weight": [None, "balanced"]}},
            {"name": "Random forest", "estimator": RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state), "dense": False,
             "params": {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5], "model__max_features": ["sqrt", 0.7]}},
            {"name": "Extra trees", "estimator": ExtraTreesClassifier(n_estimators=250, class_weight="balanced", n_jobs=-1, random_state=random_state), "dense": False,
             "params": {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5], "model__max_features": ["sqrt", 0.7]}},
            {"name": "Decision tree", "estimator": DecisionTreeClassifier(class_weight="balanced", random_state=random_state), "dense": False,
             "params": {"model__max_depth": [None, 3, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5, 10], "model__criterion": ["gini", "entropy", "log_loss"]}},
            {"name": "SVM (RBF kernel)", "estimator": SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=random_state), "dense": False,
             "params": {"model__C": loguniform(1e-2, 1e2), "model__gamma": ["scale", "auto"], "model__class_weight": [None, "balanced"]}},
            {"name": "k-nearest neighbors", "estimator": KNeighborsClassifier(n_neighbors=5), "dense": False,
             "params": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"], "model__p": [1, 2]}},
            {"name": "MLP neural network", "estimator": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=random_state), "dense": False,
             "params": {"model__hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)}},
        ]
        if allow_dense:
            specs.extend([
                {"name": "Gaussian naive Bayes", "estimator": GaussianNB(), "dense": True,
                 "params": {"model__var_smoothing": loguniform(1e-12, 1e-7)}},
                {"name": "Linear discriminant analysis", "estimator": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"), "dense": True,
                 "params": {}},
                {"name": "Gradient boosting", "estimator": GradientBoostingClassifier(random_state=random_state), "dense": True,
                 "params": {"model__n_estimators": [75, 150, 250], "model__learning_rate": [0.02, 0.05, 0.1], "model__max_depth": [1, 2, 3]}},
                {"name": "Histogram gradient boosting", "estimator": HistGradientBoostingClassifier(random_state=random_state), "dense": True,
                 "params": {"model__max_iter": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [15, 31]}},
                {"name": "AdaBoost", "estimator": AdaBoostClassifier(random_state=random_state), "dense": True,
                 "params": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.1, 1.0]}},
            ])
        return specs

    specs = [
        {"name": "Dummy baseline", "estimator": DummyRegressor(strategy="mean"), "dense": False, "params": {}},
        {"name": "Linear regression", "estimator": LinearRegression(), "dense": False, "params": {}},
        {"name": "Ridge regression", "estimator": Ridge(), "dense": False,
         "params": {"model__alpha": loguniform(1e-3, 1e3)}},
        {"name": "Lasso regression", "estimator": Lasso(max_iter=5000), "dense": False,
         "params": {"model__alpha": loguniform(1e-4, 1e1)}},
        {"name": "Elastic net", "estimator": ElasticNet(max_iter=5000), "dense": False,
         "params": {"model__alpha": loguniform(1e-4, 1e1), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}},
        {"name": "Random forest", "estimator": RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=random_state), "dense": False,
         "params": {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5], "model__max_features": [1.0, 0.7, "sqrt"]}},
        {"name": "Extra trees", "estimator": ExtraTreesRegressor(n_estimators=250, n_jobs=-1, random_state=random_state), "dense": False,
         "params": {"model__n_estimators": [150, 250, 400], "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5], "model__max_features": [1.0, 0.7, "sqrt"]}},
        {"name": "Decision tree", "estimator": DecisionTreeRegressor(random_state=random_state), "dense": False,
         "params": {"model__max_depth": [None, 3, 5, 10, 20], "model__min_samples_leaf": [1, 2, 5, 10]}},
        {"name": "Support vector regression", "estimator": SVR(kernel="rbf"), "dense": False,
         "params": {"model__C": loguniform(1e-2, 1e3), "model__epsilon": [0.01, 0.1, 0.2], "model__gamma": ["scale", "auto"]}},
        {"name": "k-nearest neighbors", "estimator": KNeighborsRegressor(n_neighbors=5), "dense": False,
         "params": {"model__n_neighbors": [3, 5, 7, 11, 15], "model__weights": ["uniform", "distance"], "model__p": [1, 2]}},
        {"name": "MLP neural network", "estimator": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=random_state), "dense": False,
         "params": {"model__hidden_layer_sizes": [(32,), (64, 32), (128, 64)], "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)}},
    ]
    if allow_dense:
        specs.extend([
            {"name": "Gradient boosting", "estimator": GradientBoostingRegressor(random_state=random_state), "dense": True,
             "params": {"model__n_estimators": [75, 150, 250], "model__learning_rate": [0.02, 0.05, 0.1], "model__max_depth": [1, 2, 3]}},
            {"name": "Histogram gradient boosting", "estimator": HistGradientBoostingRegressor(random_state=random_state), "dense": True,
             "params": {"model__max_iter": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_leaf_nodes": [15, 31]}},
            {"name": "AdaBoost", "estimator": AdaBoostRegressor(random_state=random_state), "dense": True,
             "params": {"model__n_estimators": [50, 100, 200], "model__learning_rate": [0.03, 0.1, 1.0]}},
        ])
    return specs


def _make_pipeline(X: pd.DataFrame, task: str, spec: dict[str, Any], feature_selection: str,
                   k_features: int, random_state: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocess", make_preprocessor(X, dense=bool(spec["dense"])))]
    if feature_selection == "ANOVA F-score":
        score_function = f_classif if task == "classification" else f_regression
        steps.append(("select", SafeSelectKBest(score_func=score_function, k=max(1, int(k_features)))))
    elif feature_selection == "Model-based (Extra Trees)":
        selector_estimator = (
            ExtraTreesClassifier(n_estimators=120, class_weight="balanced", n_jobs=-1, random_state=random_state)
            if task == "classification"
            else ExtraTreesRegressor(n_estimators=120, n_jobs=-1, random_state=random_state)
        )
        steps.append(("select", SelectFromModel(selector_estimator, threshold=-np.inf, max_features=max(1, int(k_features)))))
    steps.append(("model", clone(spec["estimator"])))
    return Pipeline(steps)


def make_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    strategy: str = "Random (stratified for classification)",
    test_size: float = 0.20,
    groups: pd.Series | None = None,
    times: pd.Series | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a holdout split appropriate to independent, grouped or temporal data."""
    indices = np.arange(len(X))
    if strategy.startswith("Group"):
        if groups is None:
            raise ValueError("Group split selected, but no group column was supplied.")
        if groups.isna().any():
            raise ValueError("The group/split column has missing values. Correct them before a grouped split.")
        if groups.nunique() < 3:
            raise ValueError("At least three distinct groups are needed for a group holdout plus cross-validation.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups.to_numpy()))
        return train_idx, test_idx

    if strategy.startswith("Time"):
        if times is None:
            raise ValueError("Time-based split selected, but no time column was supplied.")
        parsed = pd.to_datetime(times, errors="coerce", format="mixed")
        if parsed.isna().any():
            raise ValueError("The selected time column has missing or unparseable values. Correct these before a time split.")
        order = np.argsort(parsed.to_numpy(), kind="stable")
        n_test = max(1, int(math.ceil(len(order) * test_size)))
        if len(order) - n_test < 5:
            raise ValueError("Too few training rows remain for time-aware cross-validation. Use more observations or a smaller test fraction.")
        # Keep identical timestamps together to avoid splitting same-time rows across the boundary.
        cut = len(order) - n_test
        boundary = parsed.iloc[order[cut]]
        train_idx = np.flatnonzero((parsed < boundary).to_numpy())
        test_idx = np.flatnonzero((parsed >= boundary).to_numpy())
        if len(train_idx) < 5 or len(test_idx) < 1:
            train_idx, test_idx = order[:cut], order[cut:]
        train_idx = train_idx[np.argsort(parsed.iloc[train_idx].to_numpy(), kind="stable")]
        test_idx = test_idx[np.argsort(parsed.iloc[test_idx].to_numpy(), kind="stable")]
        return train_idx, test_idx

    stratify = None
    if task == "classification":
        class_counts = y.value_counts(dropna=False)
        if len(class_counts) < 2:
            raise ValueError("Classification requires at least two target classes.")
        if class_counts.min() < 2:
            raise ValueError("At least one target class has fewer than two rows; a stratified holdout cannot be formed reliably.")
        stratify = y
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=stratify,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def _make_cv_splits(X_train: pd.DataFrame, y_train: pd.Series, task: str, strategy: str,
                    groups_train: pd.Series | None, times_train: pd.Series | None, requested_splits: int,
                    random_state: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    if strategy.startswith("Group"):
        if groups_train is None:
            raise ValueError("Group cross-validation requires the selected group column.")
        n_splits = min(requested_splits, int(groups_train.nunique()))
        if n_splits < 2:
            raise ValueError("At least two distinct training groups are required for cross-validation.")
        if task == "classification":
            try:
                from sklearn.model_selection import StratifiedGroupKFold
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                splits = list(splitter.split(X_train, y_train, groups_train))
            except (ImportError, ValueError):
                splits = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups_train))
        else:
            splits = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups_train))
        return splits, n_splits

    if strategy.startswith("Time"):
        n_splits = min(requested_splits, len(X_train) - 2)
        if n_splits < 2:
            raise ValueError("At least four training observations are required for time-series cross-validation.")
        if times_train is not None:
            parsed = pd.to_datetime(times_train, errors="coerce", format="mixed")
            order = np.argsort(parsed.to_numpy(), kind="stable")
            X_ordered = np.arange(len(X_train))[order]
            splits = []
            for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X_ordered):
                splits.append((X_ordered[tr], X_ordered[va]))
            return splits, n_splits
        return list(TimeSeriesSplit(n_splits=n_splits).split(X_train)), n_splits

    if task == "classification":
        class_counts = y_train.value_counts()
        n_splits = min(requested_splits, int(class_counts.min()))
        if n_splits < 2:
            raise ValueError("At least two examples of every target class are needed in the training partition for stratified cross-validation.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(X_train, y_train)), n_splits
    n_splits = min(requested_splits, max(2, len(X_train) - 1))
    if len(X_train) < 4 or n_splits < 2:
        raise ValueError("At least four training observations are required for cross-validation.")
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X_train, y_train)), n_splits


def benchmark_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_strategy: str,
    groups: pd.Series | None = None,
    times: pd.Series | None = None,
    primary_metric: str | None = None,
    feature_selection: str = "None",
    k_features: int = 20,
    requested_cv_splits: int = 5,
    tune_top_n: int = 0,
    n_iter: int = 8,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Benchmark all candidates on training data only; reserve holdout data for evaluation."""
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True) if groups is not None else None
    times_train = times.iloc[train_idx].reset_index(drop=True) if times is not None else None
    if split_strategy.startswith("Time") and times_train is not None:
        order = np.argsort(pd.to_datetime(times_train, errors="coerce", format="mixed").to_numpy(), kind="stable")
        X_train = X_train.iloc[order].reset_index(drop=True)
        y_train = y_train.iloc[order].reset_index(drop=True)
        if groups_train is not None:
            groups_train = groups_train.iloc[order].reset_index(drop=True)
        times_train = times_train.iloc[order].reset_index(drop=True)

    n_classes = int(y_train.nunique()) if task == "classification" else 0
    metrics = metric_catalog(task, n_classes=n_classes)
    if primary_metric not in metrics:
        primary_metric = "roc_auc" if task == "classification" and n_classes == 2 else (
            "roc_auc_ovr" if task == "classification" else "rmse"
        )
        if primary_metric not in metrics:
            primary_metric = "f1_macro" if task == "classification" else "rmse"
    cv_splits, actual_cv_splits = _make_cv_splits(
        X_train, y_train, task, split_strategy, groups_train, times_train,
        requested_cv_splits, random_state,
    )
    invalid_auc_folds: list[int] = []
    if task == "classification":
        expected_classes = set(pd.unique(y_train))
        invalid_auc_folds = [fold_i + 1 for fold_i, (_, val_idx) in enumerate(cv_splits)
                             if set(pd.unique(y_train.iloc[val_idx])) != expected_classes]
        auc_metrics = {"roc_auc", "roc_auc_ovr", "average_precision"}
        if invalid_auc_folds and primary_metric in auc_metrics:
            raise ValueError(
                f"The selected {metrics[primary_metric]['label']} metric is not estimable in validation fold(s) {invalid_auc_folds} "
                "because at least one target class is absent. Use a suitable split/fold count, collect more data, or choose a metric "
                "that is valid for the observed folds; do not silently treat undefined AUC as zero."
            )
        if invalid_auc_folds:
            for auc_metric in auc_metrics - {primary_metric}:
                metrics.pop(auc_metric, None)
    scoring = {key: spec["scoring"] for key, spec in metrics.items()}
    dense_cells = estimate_dense_cells(X_train)
    allow_dense = dense_cells <= DENSE_CELL_LIMIT
    specs = _model_specs(task, random_state, allow_dense)
    skipped_dense = not allow_dense

    rows: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}
    spec_by_name: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name = spec["name"]
        pipe = _make_pipeline(X_train, task, spec, feature_selection, k_features, random_state)
        spec_by_name[name] = spec
        pipelines[name] = pipe
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                scores = cross_validate(
                    pipe, X_train, y_train, cv=cv_splits, scoring=scoring,
                    return_train_score=False, n_jobs=1, error_score="raise",
                )
            row: dict[str, Any] = {"model": name, "status": "ok", "cv_folds": actual_cv_splits,
                                   "cv_primary_metric": primary_metric,
                                   "cv_selection_score_mean": float(np.mean(scores[f"test_{primary_metric}"])),
                                   "cv_selection_score_sd": float(np.std(scores[f"test_{primary_metric}"], ddof=1)) if actual_cv_splits > 1 else 0.0,
                                   "uses_dense_matrix": bool(spec["dense"])}
            for key, metric in metrics.items():
                values = np.asarray(scores[f"test_{key}"], dtype=float)
                mean = float(np.nanmean(values))
                sd = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
                # Convert sklearn's negative loss scorer back to a positive metric for display.
                if metric["direction"] == "min":
                    mean, sd = -mean, sd
                row[f"cv_{key}_mean"] = mean
                row[f"cv_{key}_sd"] = sd
            warning_messages = sorted({str(w.message)[:180] for w in caught})
            row["cv_primary_mean"] = row.get(f"cv_{primary_metric}_mean", np.nan)
            row["cv_primary_sd"] = row.get(f"cv_{primary_metric}_sd", np.nan)
            row["warnings"] = " | ".join(warning_messages[:3])
            rows.append(row)
        except Exception as exc:  # model-specific failures should not block all remaining candidates
            rows.append({"model": name, "status": "failed", "cv_folds": actual_cv_splits,
                         "cv_primary_metric": primary_metric, "cv_selection_score_mean": np.nan,
                         "cv_selection_score_sd": np.nan, "uses_dense_matrix": bool(spec["dense"]),
                         "error": f"{type(exc).__name__}: {str(exc)[:350]}", "warnings": ""})

    table = pd.DataFrame(rows)
    successful = table.loc[table["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("All models failed. Review data types, sample size, target encoding, and the error details in the benchmark table.")
    table = table.sort_values("cv_selection_score_mean", ascending=False, na_position="last").reset_index(drop=True)

    tuning_notes: list[str] = []
    tuned_estimators: dict[str, Pipeline] = {}
    finalists: list[str] = []
    if tune_top_n > 0:
        finalists = table.loc[table["status"] == "ok", "model"].head(max(1, int(tune_top_n))).tolist()
        table["tuning_finalist"] = table["model"].isin(finalists)
        for name in finalists:
            param_distributions = spec_by_name[name]["params"]
            if not param_distributions:
                tuning_notes.append(f"{name}: no tunable parameters defined; used default settings.")
                continue
            search = RandomizedSearchCV(
                estimator=pipelines[name],
                param_distributions=param_distributions,
                n_iter=max(1, int(n_iter)),
                scoring=metrics[primary_metric]["scoring"],
                cv=cv_splits,
                refit=True,
                random_state=random_state,
                n_jobs=1,
                error_score=np.nan,
                return_train_score=False,
            )
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    search.fit(X_train, y_train)
                if not np.isfinite(search.best_score_):
                    tuning_notes.append(f"{name}: tuning produced no finite score; retained default CV result.")
                    continue
                tuned_estimators[name] = search.best_estimator_
                tuned_score = float(search.best_score_)
                table.loc[table["model"] == name, "cv_selection_score_mean"] = tuned_score
                tuned_display_score = -tuned_score if metrics[primary_metric]["direction"] == "min" else tuned_score
                table.loc[table["model"] == name, "cv_primary_mean"] = tuned_display_score
                table.loc[table["model"] == name, f"cv_{primary_metric}_mean"] = tuned_display_score
                best_std = float(search.cv_results_["std_test_score"][search.best_index_])
                table.loc[table["model"] == name, "cv_primary_sd"] = best_std
                table.loc[table["model"] == name, f"cv_{primary_metric}_sd"] = best_std
                table.loc[table["model"] == name, "tuned"] = True
                table.loc[table["model"] == name, "best_parameters"] = [str(search.best_params_)]
                table.loc[table["model"] == name, "tuning_warnings"] = [" | ".join(sorted({str(w.message)[:160] for w in caught})[:2])]
                display_score = -tuned_score if metrics[primary_metric]["direction"] == "min" else tuned_score
                tuning_notes.append(f"{name}: randomized search completed ({n_iter} iteration(s)); best {metrics[primary_metric]['label']}={display_score:.4f}.")
            except Exception as exc:
                tuning_notes.append(f"{name}: tuning failed ({type(exc).__name__}: {str(exc)[:180]}); default CV result retained.")

    table = table.sort_values("cv_selection_score_mean", ascending=False, na_position="last").reset_index(drop=True)
    if tune_top_n > 0 and finalists:
        # Only compare the finalists after tuning; comparing tuned finalist scores to untuned
        # scores from other algorithms would not be an apples-to-apples selection rule.
        selection_table = table.loc[table["model"].isin(finalists)].sort_values(
            "cv_selection_score_mean", ascending=False, na_position="last"
        )
        best_name = str(selection_table.iloc[0]["model"])
    else:
        best_name = str(table.iloc[0]["model"])
    table["selected_for_final_test"] = table["model"] == best_name
    if tune_top_n > 0 and finalists:
        table["selection_stage"] = table["model"].map(
            lambda name: "Tuned finalist" if name in tuned_estimators else (
                "Finalist; default parameters" if name in finalists else "Initial screen only"
            )
        )
        table["_finalist_order"] = table["model"].apply(lambda name: 0 if name in finalists else 1)
        table = table.sort_values(
            ["selected_for_final_test", "_finalist_order", "cv_selection_score_mean"],
            ascending=[False, True, False], na_position="last",
        ).drop(columns="_finalist_order").reset_index(drop=True)
    else:
        table["selection_stage"] = "Initial CV screen"
        table = table.sort_values("cv_selection_score_mean", ascending=False, na_position="last").reset_index(drop=True)
    best_pipeline = tuned_estimators.get(best_name, pipelines[best_name])
    # Save the training rows in their exact CV order if time-aware sorting was applied.
    training_index_order = np.asarray(train_idx)
    if split_strategy.startswith("Time") and times is not None:
        training_index_order = training_index_order[np.argsort(
            pd.to_datetime(times.iloc[training_index_order], errors="coerce", format="mixed").to_numpy(), kind="stable")]
    return {
        "cv_table": table,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "train_idx": training_index_order,
        "test_idx": np.asarray(test_idx),
        "primary_metric": primary_metric,
        "metric_spec": metrics[primary_metric],
        "all_metrics": metrics,
        "actual_cv_splits": actual_cv_splits,
        "dense_cells_estimate": dense_cells,
        "dense_models_skipped": skipped_dense,
        "tuning_notes": tuning_notes,
        "tuning_finalists": finalists,
        "training_X": X.iloc[training_index_order].reset_index(drop=True),
        "training_y": y.iloc[training_index_order].reset_index(drop=True),
    }


def evaluate_final_model(
    benchmark: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    primary_metric: str | None = None,
    random_state: int = RANDOM_STATE,
    permutation_repeats: int = 5,
    row_ids: list[Any] | np.ndarray | None = None,
    class_label_mapping: dict[Any, Any] | None = None,
) -> dict[str, Any]:
    """Fit the selected pipeline on all training rows and evaluate once on the holdout."""
    from sklearn.inspection import permutation_importance
    train_idx = np.asarray(benchmark["train_idx"])
    test_idx = np.asarray(benchmark["test_idx"])
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    model = benchmark["best_pipeline"]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics: dict[str, Any] = {}
    report_df = pd.DataFrame()
    confusion_df = pd.DataFrame()
    if task == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test, predictions))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, predictions))
        metrics["precision_macro"] = float(precision_score(y_test, predictions, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_test, predictions, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_test, predictions, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_test, predictions, average="weighted", zero_division=0))
        if class_label_mapping:
            labels = sorted(class_label_mapping.keys())
        else:
            labels = list(pd.unique(pd.concat([y_train, y_test], ignore_index=True)))
        display_labels = [str((class_label_mapping or {}).get(label, label)) for label in labels]
        report_df = pd.DataFrame(classification_report(y_test, predictions, labels=labels, target_names=display_labels, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "class_or_average"})
        confusion_df = pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=[f"actual_{x}" for x in display_labels], columns=[f"predicted_{x}" for x in display_labels]).reset_index(names="actual_class")
        score_vector = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if len(model.classes_) == 2:
                    score_vector = proba[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, score_vector))
                    metrics["average_precision"] = float(average_precision_score(y_test, score_vector, pos_label=model.classes_[1]))
                    metrics["brier_score"] = float(np.mean((score_vector - (y_test.to_numpy() == model.classes_[1]).astype(float)) ** 2))
                else:
                    metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_test, proba, labels=model.classes_, multi_class="ovr", average="macro"))
            elif hasattr(model, "decision_function"):
                decision = model.decision_function(X_test)
                if len(model.classes_) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, decision))
                else:
                    metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_test, decision, labels=model.classes_, multi_class="ovr", average="macro"))
        except Exception as exc:
            metrics["auc_note"] = f"AUC was not estimable on this holdout: {type(exc).__name__}: {str(exc)[:180]}"
        actual_display = [((class_label_mapping or {}).get(v, v)) for v in y_test.to_numpy()]
        predicted_display = [((class_label_mapping or {}).get(v, v)) for v in predictions]
        source_ids = np.asarray(row_ids, dtype=object)[test_idx] if row_ids is not None else test_idx
        predictions_df = pd.DataFrame({
            "source_row_number": source_ids,
            "actual": actual_display,
            "predicted": predicted_display,
        })
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    shown_class = (class_label_mapping or {}).get(model.classes_[1], model.classes_[1])
                    predictions_df[f"probability_class_{shown_class}"] = proba[:, 1]
                else:
                    for i, cls in enumerate(model.classes_):
                        shown_class = (class_label_mapping or {}).get(cls, cls)
                        predictions_df[f"probability_class_{shown_class}"] = proba[:, i]
            except Exception:
                pass
    else:
        residuals = y_test.to_numpy(dtype=float) - np.asarray(predictions, dtype=float)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        metrics["rmse"] = rmse
        metrics["mae"] = float(mean_absolute_error(y_test, predictions))
        metrics["r2"] = float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan")
        nonzero_mask = y_test.to_numpy(dtype=float) != 0
        if nonzero_mask.any():
            metrics["mape_nonzero_targets"] = float(mean_absolute_percentage_error(y_test.to_numpy(dtype=float)[nonzero_mask], np.asarray(predictions, dtype=float)[nonzero_mask]))
        source_ids = np.asarray(row_ids, dtype=object)[test_idx] if row_ids is not None else test_idx
        predictions_df = pd.DataFrame({
            "source_row_number": source_ids,
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "residual_actual_minus_predicted": residuals,
        })

    importance_df = pd.DataFrame()
    try:
        scorer_name = benchmark["metric_spec"]["scoring"]
        perm = permutation_importance(
            model, X_test, y_test, scoring=scorer_name,
            n_repeats=max(2, int(permutation_repeats)), random_state=random_state, n_jobs=1,
        )
        importance_df = pd.DataFrame({
            "feature": X.columns.astype(str),
            "permutation_importance_mean_score_drop": perm.importances_mean,
            "permutation_importance_sd": perm.importances_std,
        }).sort_values("permutation_importance_mean_score_drop", ascending=False).reset_index(drop=True)
    except Exception as exc:
        importance_df = pd.DataFrame([{"feature": "Importance calculation failed", "permutation_importance_mean_score_drop": np.nan, "permutation_importance_sd": np.nan, "note": f"{type(exc).__name__}: {str(exc)[:180]}"}])

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions_df,
        "classification_report": report_df,
        "confusion_matrix": confusion_df,
        "permutation_importance": importance_df,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }
