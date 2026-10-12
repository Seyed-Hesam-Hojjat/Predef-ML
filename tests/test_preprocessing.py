import numpy as np
import pandas as pd

from ml_workbench.preprocessing import prepare_features


def test_prepare_features_drops_missing_target_engineers_dates_and_converts_numeric_text():
    df = pd.DataFrame({
        "numeric_text": ["1.2", "2.4", "3.3", "4.5", "5.6"],
        "visit_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "empty": [np.nan] * 5,
        "target": [0, 1, 0, 1, np.nan],
    })
    X, y, metadata = prepare_features(df, "target", date_columns=["visit_date"])
    assert len(X) == 4
    assert len(y) == 4
    assert "visit_date" not in X.columns
    assert "visit_date__year" in metadata["engineered_date_features"]
    assert "visit_date__day" in X.columns
    assert "numeric_text" in metadata["numeric_text_columns_converted"]
    assert "empty" in metadata["all_missing_features_removed"]
    assert metadata["rows_without_target_removed"] == 1
