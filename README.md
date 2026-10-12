<<<<<<< HEAD
# Tabular ML Workbench

A local-first, interactive Python app for a guided, auditable machine-learning workflow on tabular Excel/CSV data. It is intended as a starting point for a public GitHub project and for research/analytics workflows—not as a push-button substitute for domain expertise or a prespecified analysis plan.

## What it does

1. **Import and preview** `.xlsx`, `.xlsm`, `.xls`, and `.csv` files.
2. **Audit data quality**: missing values, duplicate rows/column names, constant and near-constant features, numeric-looking text, identifier-like names, high-cardinality text, infinite values, rare target classes, and possible target-leakage candidates. It reports recommended actions rather than silently modifying the source.
3. **Confirm the target and task type**: classification or regression. For binary classification, explicitly choose the positive/event class; it is coded as 1 for binary probability metrics and decoded back to the original label in predictions. The task suggestion is a hint, not an automatic scientific decision.
4. **Prepare features**: optionally expand selected date columns into calendar components; coerce clearly numeric text fields; handle infinite values and wholly missing predictors; exclude chosen identifiers, grouping keys, time keys, or variables unavailable at prediction time.
5. **Create a deliberate holdout split**: random (stratified classification), group-based, or time-based. Group/time split columns are not used as predictors by default.
6. **Compare algorithms by cross-validation on training data only** and optionally run randomized hyperparameter searches on the top three candidates.
7. **Apply fold-fitted preprocessing**: numeric median imputation and scaling, categorical imputation and one-hot encoding, with optional ANOVA F-score or model-based Extra Trees feature selection inside the model pipeline.
8. **Evaluate the selected model once on the held-out test set**, reporting appropriate classification/regression metrics, class-wise results/confusion matrix (classification), predictions, and permutation feature importance.
9. **Export** a multi-sheet Excel report and the fitted scikit-learn pipeline.

## Included algorithm families

The exact set depends on the selected task and a memory-safety estimate. Classification candidates include dummy baseline, logistic regression, ridge classifier, random forest, extra trees, decision tree, RBF support vector machine, k-nearest neighbors, multilayer perceptron, and—when a dense transformed feature matrix is estimated to fit safely—Gaussian naive Bayes, linear discriminant analysis, gradient boosting, histogram gradient boosting, and AdaBoost. Regression candidates include dummy baseline, linear regression, ridge, lasso, elastic net, random forest, extra trees, decision tree, support vector regression, k-nearest neighbors, multilayer perceptron, and—when safe—gradient boosting, histogram gradient boosting, and AdaBoost.

The intent is a set of common tabular baselines and algorithms, not a claim that every possible algorithm is appropriate for every dataset. External libraries such as XGBoost, LightGBM, CatBoost, imbalanced-learn, survival models, and AutoML frameworks are not required dependencies; they can be integrated later with method-specific validation.

## Quick start

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal and upload an Excel/CSV file. A small synthetic dataset is included at `sample_data/synthetic_classification.csv` for a smoke test.

## Recommended workflow

- Use **one row per independent prediction unit** whenever that reflects the data-generating process.
- Select a target that reflects the intended prediction question. Check that every predictor would be available at the moment the prediction is made.
- For repeated records from the same person, clinic, hospital, family, or other unit, use **group-based splitting** when those groups must not cross train/test boundaries.
- For forecasting or future deployment, use **time-based splitting** and respect the order of observations. Standard random splits are inappropriate for many temporal tasks.
- Predefine a primary metric that reflects the costs of prediction errors. For imbalanced binary classification, ROC AUC alone may be insufficient; consider precision-recall, sensitivity/recall, specificity, calibration, and decision-analytic utility.
- Keep the test partition untouched until the design and model-selection procedure are fixed. If you change features, hyperparameters, or decisions in response to test performance, the test set is no longer an untouched final test.
- Report uncertainty. A single test-set score is not a confidence interval and may be unstable in small datasets. Consider bootstrap confidence intervals, nested cross-validation, and external validation for serious scientific claims.

## Methodological notes and limitations

### Leakage and preprocessing

Imputation, scaling, one-hot encoding, feature selection, and the estimator are combined in scikit-learn pipelines. Those learned transformations are refit within each training fold, limiting preprocessing leakage. The optional date-component expansion is deterministic. Name-based audit rules are only heuristics: they cannot identify all leakage, and they may flag legitimate fields. Human review of the prediction-time data flow is essential.

### Model-selection bias

The app first screens models using training-only cross-validation. Optional tuning applies randomized search to the top three screening candidates and selects among those tuned finalists using training-only CV results; it does not compare their tuned scores against untouched algorithms' untuned scores. The selected pipeline is then assessed on a held-out test partition. Screening the algorithms and hyperparameters on the same training data can still make cross-validation comparisons optimistic. For high-stakes or confirmatory claims, plan nested cross-validation and/or external validation.

### Not supported automatically

This generic program is not a dedicated solution for survival/time-to-event analysis, competing risks, repeated measures without a valid group split, clustered outcomes needing multilevel models, causal inference, images, audio, complex NLP, multi-label classification, censored data, or large-scale sparse text. Those require specific data structures, validation strategies, metrics, and model families. Categorical integer coding is inferred heuristically and should be reviewed.

The workflow does not currently provide an automated fairness audit, calibration curve/report, threshold optimization, bootstrap confidence intervals, feature-attribution uncertainty, or formal cost-effectiveness analysis. Those are appropriate future contributions.

### Data privacy

The app runs locally by default and the code does not intentionally transmit datasets to a remote service. Still, do not upload identifiable, sensitive, or regulated data unless your local environment, storage, and institutional policies permit the processing. Streamlit session state may reset on browser reload.

### Export safety

The downloadable `.joblib` is a Python serialized predictor. As with pickle-based formats, **loading an untrusted file may execute code**. Only load artifacts you created or otherwise trust. Use the same compatible Python and package versions when loading the model. After loading a trusted artifact, call `predictor.predict(new_dataframe)` and, for models that support it, `predictor.predict_proba(new_dataframe)`. The new DataFrame should contain the raw predictor columns used during training; absent predictors are filled as missing for the fitted imputers, and extra columns are ignored.

## Repository layout

```text
app.py                       # Guided Streamlit UI
ml_workbench/audit.py        # Data audit and warning rules
ml_workbench/preprocessing.py# Feature preparation and sklearn preprocessing
ml_workbench/modeling.py     # CV, algorithms, split logic, final evaluation
ml_workbench/inference.py    # Raw-input prediction wrapper for exported models
ml_workbench/reporting.py    # Excel and model artifact exports
tests/                       # Unit/smoke tests
sample_data/                 # Synthetic demo data only
```

## Contributing roadmap

Potential next steps include nested CV, repeated-CV uncertainty summaries, bootstrap confidence intervals, calibration/decision curves, threshold selection on validation data, SHAP or other explainability tools, fairness checks, optional XGBoost/LightGBM/CatBoost integrations, a command-line interface, API endpoints, and a reproducibility manifest for each run.

## License

MIT. See `LICENSE`.

## Technical references

- scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html), including train/test separation and preprocessing leakage.
- scikit-learn, [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html), for fitting transformations within validation folds.
- scikit-learn, [Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), for scoring and evaluation metrics.
- scikit-learn, [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html), including its interpretation limitations.
- Streamlit, [Session State](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state), for the multi-step local interface.
=======
# Predef-ML
>>>>>>> 723f3b963f48856be4bf1c5749eb0380175154a1
