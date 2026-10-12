"""Data-quality checks and actionable warnings for tabular ML projects."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

ID_NAME_RE = re.compile(r"(^id$|(^|_)id($|_)|uuid|record.?number|patient.?number|mrn|email)", re.I)
LEAKAGE_NAME_RE = re.compile(
    r"(outcome|target|label|final.?diagnos|discharg|death|mortality|readmi|complication.?after|post.?treatment|response.?to.?treatment)",
    re.I,
)
DATE_NAME_RE = re.compile(r"(date|datetime|timestamp|_time$|^time$|birth|admission|visit)", re.I)


def suggest_task_type(y: pd.Series) -> str:
    """Provide a conservative hint only; the user must confirm the task type."""
    non_null = y.dropna()
    if non_null.empty:
        return "classification"
    if (
        pd.api.types.is_bool_dtype(non_null)
        or pd.api.types.is_object_dtype(non_null)
        or isinstance(non_null.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(non_null)
    ):
        return "classification"
    unique = non_null.nunique(dropna=True)
    if unique <= min(10, max(2, int(np.sqrt(len(non_null))))) and unique <= 20:
        return "classification"
    return "regression"


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Suggest date-like columns based on dtype/name and a bounded parse sample."""
    result: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or DATE_NAME_RE.search(str(col)):
            # The name/dtype is sufficient to suggest; invalid values are reported separately.
            result.append(str(col))
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(100)
            if len(sample) >= 10:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() >= 0.85:
                    result.append(str(col))
    return result


def audit_dataframe(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return an actionable issue table and a compact numeric audit summary."""
    issues: list[dict[str, str]] = []

    def add(severity: str, code: str, message: str, recommendation: str, columns: str = "") -> None:
        issues.append({
            "severity": severity,
            "check": code,
            "finding": message,
            "recommended_action": recommendation,
            "affected_columns": columns,
        })

    if df.empty:
        add("CRITICAL", "EMPTY_DATA", "The uploaded dataset has no rows.", "Upload a worksheet containing observations.")
    if len(df.columns) == 0:
        add("CRITICAL", "NO_COLUMNS", "The uploaded dataset has no columns.", "Upload a worksheet containing variables.")

    duplicate_names = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].astype(str).tolist()
    if duplicate_names:
        add("HIGH", "DUPLICATE_COLUMN_NAMES", f"Duplicate column names detected: {duplicate_names}.",
            "Rename duplicate columns so every variable has a unique name.", ", ".join(duplicate_names))

    duplicated_rows = int(df.duplicated().sum()) if len(df) else 0
    if duplicated_rows:
        add("MEDIUM", "DUPLICATE_ROWS", f"{duplicated_rows} fully duplicated row(s) detected.",
            "Verify whether these are genuine repeated observations. Remove only if duplicates are not valid records.")

    all_missing = [str(c) for c in df.columns if df[c].isna().all()]
    if all_missing:
        add("HIGH", "ALL_MISSING", "Columns contain no observed values.",
            "Remove these columns or repair the source before modelling.", ", ".join(all_missing))

    n = max(1, len(df))
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct >= 40].index.astype(str).tolist()
    moderate_missing = missing_pct[(missing_pct >= 5) & (missing_pct < 40)].index.astype(str).tolist()
    if high_missing:
        add("HIGH", "HIGH_MISSINGNESS", "At least 40% of values are missing in one or more columns.",
            "Check source-system coding and whether missingness is informative; use training-fold imputation and consider a justified exclusion.",
            ", ".join(high_missing))
    if moderate_missing:
        add("MEDIUM", "MISSING_VALUES", "Missing values occur in one or more columns.",
            "Inspect missingness patterns. Numeric and categorical imputers are included in the modelling pipeline.",
            ", ".join(moderate_missing))

    constant = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant:
        add("MEDIUM", "CONSTANT_COLUMNS", "One or more columns have zero or one observed unique value.",
            "Remove constant features; an all-missing target must be corrected before proceeding.", ", ".join(constant))

    near_constant: list[str] = []
    identifier_like: list[str] = []
    high_cardinality: list[str] = []
    suspicious_leakage: list[str] = []
    numeric_text: list[str] = []
    infinity_cols: list[str] = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if len(non_null) and non_null.nunique() > 1:
            top_share = non_null.value_counts(normalize=True, dropna=True).iloc[0]
            if top_share >= 0.98:
                near_constant.append(str(col))
        if ID_NAME_RE.search(str(col)):
            identifier_like.append(str(col))
        if LEAKAGE_NAME_RE.search(str(col)) and str(col) != str(target):
            suspicious_leakage.append(str(col))
        if (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) and len(non_null) >= 20 and not DATE_NAME_RE.search(str(col)):
            unique_ratio = non_null.nunique() / max(1, len(non_null))
            if non_null.nunique() >= 20 and unique_ratio >= 0.5:
                high_cardinality.append(str(col))
            numeric_parse = pd.to_numeric(non_null.astype(str).str.replace(",", "", regex=False), errors="coerce")
            if len(non_null) and numeric_parse.notna().mean() >= 0.90 and not DATE_NAME_RE.search(str(col)):
                numeric_text.append(str(col))
        if pd.api.types.is_numeric_dtype(s):
            try:
                arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(arr).any():
                    infinity_cols.append(str(col))
            except (TypeError, ValueError):
                pass

    if near_constant:
        add("LOW", "NEAR_CONSTANT", "At least 98% of non-missing values are identical in one or more columns.",
            "Verify these features; they may add little information, but do not remove them without checking their meaning.",
            ", ".join(near_constant))
    if identifier_like:
        add("HIGH", "IDENTIFIER_CANDIDATES", "Column names suggest row identifiers or direct identifiers.",
            "Exclude identifiers from predictors unless there is a defensible, deployment-valid reason to use them. Check for person-level overlap across rows.",
            ", ".join(identifier_like))
    if high_cardinality:
        add("MEDIUM", "HIGH_CARDINALITY_TEXT", "Some text-like columns have many unique values and may behave like IDs or free text.",
            "Inspect them manually. Do not one-hot encode unrestricted identifiers; use a justified text/NLP approach if the content is meaningful.",
            ", ".join(high_cardinality))
    if suspicious_leakage:
        add("HIGH", "POSSIBLE_TARGET_LEAKAGE", "Column names suggest an outcome or information recorded after the prediction time.",
            "Confirm when each value becomes available. Exclude variables unavailable at the intended prediction time; this is a name-based warning, not proof of leakage.",
            ", ".join(suspicious_leakage))
    if numeric_text:
        add("LOW", "NUMERIC_STORED_AS_TEXT", "Some text columns appear to contain numeric values.",
            "Check decimal/thousands separators and convert these columns to numeric in the source or during data cleaning.",
            ", ".join(numeric_text))
    if infinity_cols:
        add("HIGH", "INFINITE_VALUES", "Numeric columns contain positive or negative infinity.",
            "Replace infinities with missing values or correct the calculation before modelling.", ", ".join(infinity_cols))

    if target is not None:
        if target not in df.columns:
            add("CRITICAL", "TARGET_NOT_FOUND", f"Selected target '{target}' is not present in the data.",
                "Choose an existing column as the outcome.")
        else:
            y = df[target]
            if y.isna().any():
                add("MEDIUM", "MISSING_TARGET", f"The target has {int(y.isna().sum())} missing row(s).",
                    "Rows without a known outcome are excluded from supervised modelling; verify whether missingness is systematic.", str(target))
            if y.dropna().nunique() < 2:
                add("CRITICAL", "TARGET_ONE_CLASS", "The target has fewer than two observed unique values.",
                    "A supervised model cannot learn a useful distinction from a constant target. Correct the target or select another column.", str(target))
            if len(y.dropna()) < 30:
                add("HIGH", "SMALL_SAMPLE", f"Only {len(y.dropna())} rows have an observed target.",
                    "Treat results as exploratory; cross-validation estimates will be unstable. Consider whether model complexity and the number of candidate models are justified.", str(target))
            if len(df.columns) > max(1, len(y.dropna())):
                add("HIGH", "P_MORE_THAN_N", "The number of columns exceeds the number of labelled observations.",
                    "Use strong regularization and nested resampling where feasible; report uncertainty and avoid interpreting the highest score as reliable evidence.")
            target_unique = int(y.dropna().nunique())
            classification_like_target = (
                pd.api.types.is_bool_dtype(y)
                or not pd.api.types.is_numeric_dtype(y)
                or target_unique <= 10
            )
            if classification_like_target:
                counts = y.dropna().value_counts()
                rare = counts[counts < 5]
                if len(rare):
                    add("HIGH", "RARE_TARGET_CLASSES", "At least one likely target class has fewer than five observations.",
                        "A stable stratified holdout/CV estimate may be impossible. Consider more data or a scientifically justified target definition; do not merge classes solely to improve metrics.",
                        ", ".join(map(str, rare.index.tolist())))

    report = pd.DataFrame(issues, columns=["severity", "check", "finding", "recommended_action", "affected_columns"])
    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicated_rows,
        "total_missing_cells": int(df.isna().sum().sum()),
        "missing_cell_pct": float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1]) * 100),
        "duplicate_column_names": duplicate_names,
        "all_missing_columns": all_missing,
        "date_suggestions": detect_date_columns(df),
        "identifier_candidates": identifier_like,
        "possible_leakage_candidates": suspicious_leakage,
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
    }
    return report, summary
