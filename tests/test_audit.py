import pandas as pd

from ml_workbench.audit import audit_dataframe, suggest_task_type


def test_audit_detects_missing_duplicate_ids_and_possible_leakage():
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4],
        "age": [30, None, 40, 40],
        "discharge_outcome": [0, 1, 0, 1],
    })
    issues, summary = audit_dataframe(df, target="age")
    checks = set(issues["check"])
    assert "IDENTIFIER_CANDIDATES" in checks
    assert "MISSING_VALUES" in checks
    assert "POSSIBLE_TARGET_LEAKAGE" in checks
    assert summary["rows"] == 4


def test_task_suggestion_is_only_a_hint():
    assert suggest_task_type(pd.Series(["yes", "no", "yes"])) == "classification"
    assert suggest_task_type(pd.Series([1.2, 5.4, 3.1, 9.8])) == "regression"
