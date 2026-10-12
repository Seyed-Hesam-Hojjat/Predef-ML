"""Inference wrapper that applies deterministic feature engineering to new raw data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import prepare_prediction_features


class TabularPredictor:
    """Fitted predictor that accepts raw tabular rows and returns user-facing labels.

    The wrapped estimator is a fitted scikit-learn Pipeline. This class handles only
    deterministic input preparation; learned imputers, scalers, encoders, feature selectors,
    and model parameters remain in the pipeline.
    """

    def __init__(
        self,
        estimator: Any,
        target_column: str,
        excluded_columns: list[str],
        date_columns: list[str],
        numeric_columns: list[str],
        expected_columns: list[str],
        class_label_mapping: dict[Any, Any] | None = None,
    ) -> None:
        self.estimator = estimator
        self.target_column = target_column
        self.excluded_columns = list(excluded_columns)
        self.date_columns = list(date_columns)
        self.numeric_columns = list(numeric_columns)
        self.expected_columns = list(expected_columns)
        self.class_label_mapping = dict(class_label_mapping or {})

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        return prepare_prediction_features(
            X,
            target=self.target_column,
            excluded_columns=self.excluded_columns,
            date_columns=self.date_columns,
            numeric_columns=self.numeric_columns,
            expected_columns=self.expected_columns,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prepared = self._prepare(X)
        predictions = self.estimator.predict(prepared)
        if self.class_label_mapping:
            return np.asarray([self.class_label_mapping.get(value, value) for value in predictions], dtype=object)
        return np.asarray(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("The selected model does not support predict_proba().")
        return self.estimator.predict_proba(self._prepare(X))

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.estimator, "decision_function"):
            raise AttributeError("The selected model does not support decision_function().")
        return self.estimator.decision_function(self._prepare(X))

    @property
    def classes_(self) -> np.ndarray:
        classes = getattr(self.estimator, "classes_", None)
        if classes is None:
            raise AttributeError("This predictor is a regression model and has no classes_.")
        if self.class_label_mapping:
            return np.asarray([self.class_label_mapping.get(value, value) for value in classes], dtype=object)
        return np.asarray(classes)

    @property
    def feature_columns_(self) -> list[str]:
        """Predictor columns expected after deterministic feature preparation."""
        return list(self.expected_columns)
