"""Leakage-conscious feature preparation and preprocessing pipelines."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(
    df: pd.DataFrame,
    target: str,
    excluded_columns: Iterable[str] | None = None,
    date_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Apply deterministic, outcome-independent cleaning and date expansion.

    Learned transformations (imputation, scaling, encoding and selection) are deliberately
    not run here; they live inside sklearn Pipelines and are fitted inside CV folds only.
    """
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} was not found.")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique. Rename duplicate columns before continuing.")

    working = df.copy()
    target_missing = int(working[target].isna().sum())
    usable_mask = working[target].notna()
    working = working.loc[usable_mask].copy()
    y = working[target].copy().reset_index(drop=True)

    excluded = set(excluded_columns or [])
    date_cols = [c for c in (date_columns or []) if c in working.columns and c != target]
    feature_cols = [c for c in working.columns if c != target and c not in excluded]
    X = working[feature_cols].copy().reset_index(drop=True)

    engineered_dates: list[str] = []
    expanded_date_columns: list[str] = []
    for col in date_cols:
        if col not in X.columns:
            continue
        parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
        # An all-invalid date column is not silently converted into useless features.
        if parsed.notna().sum() == 0:
            continue
        X[f"{col}__year"] = parsed.dt.year.astype("float64")
        X[f"{col}__month"] = parsed.dt.month.astype("float64")
        X[f"{col}__day"] = parsed.dt.day.astype("float64")
        X[f"{col}__dayofweek"] = parsed.dt.dayofweek.astype("float64")
        engineered_dates.extend([f"{col}__year", f"{col}__month", f"{col}__day", f"{col}__dayofweek"])
        X.drop(columns=[col], inplace=True)
        expanded_date_columns.append(col)

    # Coerce clearly numeric text columns, preserving fields that appear to be dates.
    converted_numeric_text: list[str] = []
    for col in list(X.columns):
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
            values = X[col].dropna().astype(str).str.replace(",", "", regex=False).str.strip()
            if len(values) < 3:
                continue
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().mean() >= 0.95:
                X[col] = pd.to_numeric(X[col].astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")
                converted_numeric_text.append(str(col))

    # Infinite values are converted to missing so the fold-local imputer can handle them.
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        X[numeric_cols] = X[numeric_cols].replace([np.inf, -np.inf], np.nan)

    all_missing_features = [str(c) for c in X.columns if X[c].isna().all()]
    if all_missing_features:
        X.drop(columns=all_missing_features, inplace=True)

    # Standardize names to strings for sklearn's ColumnTransformer and downloadable reports.
    X.columns = [str(c) for c in X.columns]
    if not X.columns.is_unique:
        raise ValueError("Two predictor column names become identical after conversion to text. Rename the source columns to unique names.")
    constant_features = [str(c) for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_features:
        X.drop(columns=constant_features, inplace=True)
    y.name = str(target)
    source_row_numbers = (np.flatnonzero(usable_mask.to_numpy()) + 2).astype(int).tolist()
    metadata: dict[str, object] = {
        "source_rows": int(len(df)),
        "labelled_rows": int(len(working)),
        "rows_without_target_removed": target_missing,
        "source_row_numbers": source_row_numbers,
        "excluded_columns": sorted(excluded),
        "date_columns_selected": date_cols,
        "date_columns_engineered": expanded_date_columns,
        "engineered_date_features": engineered_dates,
        "numeric_text_columns_converted": converted_numeric_text,
        "all_missing_features_removed": all_missing_features,
        "constant_features_removed": constant_features,
    }
    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain. Check excluded columns and select usable features.")
    return X, y, metadata


def get_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numerical versus categorical columns for tabular preprocessing."""
    numeric: list[str] = []
    categorical: list[str] = []
    for col in X.columns:
        s = X[col]
        # Low-cardinality integer/bool columns are usually codes/categories; treat as categorical.
        if pd.api.types.is_bool_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s):
            unique = s.nunique(dropna=True)
            if pd.api.types.is_integer_dtype(s) and 2 <= unique <= min(12, max(2, int(np.sqrt(max(1, len(s)))))):
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def make_preprocessor(X: pd.DataFrame, dense: bool = False) -> ColumnTransformer:
    """Create fold-fitted imputing, scaling and encoding steps.

    Dense output is only used for estimators that require it, and the UI guards its estimated
    size before those estimators are added to the benchmark.
    """
    numeric_cols, categorical_cols = get_column_types(X)
    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
                max_categories=50,
                sparse_output=not dense,
            )),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0 if not dense else 0.0)


def estimate_dense_cells(X: pd.DataFrame) -> int:
    """Conservative upper-ish estimate of dense one-hot matrix cells for safety gating."""
    numeric, categorical = get_column_types(X)
    expanded = len(numeric) * 2  # possible missing indicators
    for col in categorical:
        n_unique = int(X[col].nunique(dropna=True))
        expanded += min(max(1, n_unique), 50) + (1 if X[col].isna().any() else 0)
    return int(len(X) * max(1, expanded))


def prepare_prediction_features(
    df: pd.DataFrame,
    target: str,
    excluded_columns: Iterable[str],
    date_columns: Iterable[str],
    numeric_columns: Iterable[str],
    expected_columns: Iterable[str],
) -> pd.DataFrame:
    """Apply the same deterministic feature preparation to new raw rows.

    Learned preprocessing remains inside the saved estimator pipeline. Missing expected
    predictor columns are introduced as NaN so the fitted imputation steps can handle them.
    Extra columns (including the target, IDs, or metadata columns) are ignored.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Prediction input must be a pandas DataFrame.")
    frame = df.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError("Prediction input has duplicate column names after normalizing headers.")

    excluded = set(map(str, excluded_columns or []))
    date_cols = list(map(str, date_columns or []))
    numeric_cols = set(map(str, numeric_columns or []))
    wanted_columns = list(map(str, expected_columns or []))
    feature_cols = [c for c in frame.columns if c != str(target) and c not in excluded]
    X = frame[feature_cols].copy()

    for col in date_cols:
        if col not in X.columns:
            continue
        parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
        X[f"{col}__year"] = parsed.dt.year.astype("float64")
        X[f"{col}__month"] = parsed.dt.month.astype("float64")
        X[f"{col}__day"] = parsed.dt.day.astype("float64")
        X[f"{col}__dayofweek"] = parsed.dt.dayofweek.astype("float64")
        X.drop(columns=[col], inplace=True)

    # Cast features that were numeric during training even when an input CSV parses them as text.
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col].astype("string").str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
    for col in list(X.columns):
        if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
            values = X[col].dropna().astype(str).str.replace(",", "", regex=False).str.strip()
            if len(values) >= 3:
                parsed = pd.to_numeric(values, errors="coerce")
                if parsed.notna().mean() >= 0.95 and col in numeric_cols:
                    X[col] = pd.to_numeric(X[col].astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")
    numeric_present = [col for col in X.columns if col in numeric_cols]
    if numeric_present:
        X[numeric_present] = X[numeric_present].replace([np.inf, -np.inf], np.nan)
    X.columns = [str(c) for c in X.columns]
    X = X.reindex(columns=wanted_columns)
    return X
