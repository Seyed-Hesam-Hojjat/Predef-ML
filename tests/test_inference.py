from io import BytesIO

import joblib
import numpy as np
import pandas as pd

from ml_workbench.inference import TabularPredictor
from ml_workbench.modeling import benchmark_models, evaluate_final_model, make_holdout_split
from ml_workbench.preprocessing import prepare_features
from ml_workbench.reporting import model_bytes


def test_exported_style_predictor_accepts_raw_rows_and_decodes_labels():
    rng = np.random.default_rng(55)
    n = 90
    age = rng.normal(50, 10, n)
    date = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    group = rng.choice(["A", "B", "C"], n)
    labels = np.where(age + (group == "A") * 4 + rng.normal(0, 5, n) > 51, "event", "no_event")
    raw = pd.DataFrame({"age_text": [str(round(x, 2)) for x in age], "visit_date": date, "group": group, "outcome": labels})
    X, y, metadata = prepare_features(raw, "outcome", date_columns=["visit_date"])
    mapping = {0: "no_event", 1: "event"}
    y = y.map({"no_event": 0, "event": 1}).astype(int)
    train_idx, test_idx = make_holdout_split(X, y, "classification", test_size=0.2)
    benchmark = benchmark_models(
        X, y, "classification", train_idx, test_idx,
        "Random (stratified for classification)", primary_metric="roc_auc", requested_cv_splits=3,
    )
    evaluated = evaluate_final_model(benchmark, X, y, "classification", permutation_repeats=2)
    model = TabularPredictor(
        estimator=evaluated["model"], target_column="outcome", excluded_columns=[],
        date_columns=["visit_date"],
        numeric_columns=X.select_dtypes(include=[np.number]).columns.astype(str).tolist(),
        expected_columns=X.columns.astype(str).tolist(), class_label_mapping=mapping,
    )
    new_rows = raw.drop(columns=["outcome"]).head(5)
    prediction = model.predict(new_rows)
    probabilities = model.predict_proba(new_rows)
    assert set(prediction).issubset({"event", "no_event"})
    assert probabilities.shape == (5, 2)
    assert list(model.classes_) == ["no_event", "event"]
    restored = joblib.load(BytesIO(model_bytes(model)))
    assert np.array_equal(restored.predict(new_rows), prediction)
