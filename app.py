from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import streamlit as st

from ml_workbench.audit import audit_dataframe, detect_date_columns, suggest_task_type
from ml_workbench.modeling import benchmark_models, evaluate_final_model, make_holdout_split, metric_catalog
from ml_workbench.inference import TabularPredictor
from ml_workbench.preprocessing import prepare_features
from ml_workbench.reporting import excel_report_bytes, model_bytes

st.set_page_config(page_title="Tabular ML Workbench", page_icon="🧪", layout="wide")

st.title("Tabular ML Workbench")
st.caption("A guided, local-first workflow for transparent tabular classification and regression.")
st.markdown(
    "Upload an Excel/CSV file, audit it, confirm the prediction target and split strategy, compare algorithms by cross-validation, "
    "then evaluate one selected model on a held-out test set. Your data are processed in this app session; they are not intentionally sent to an external service."
)


def _file_fingerprint(data: bytes, sheet: str | None) -> str:
    return hashlib.sha256(data + str(sheet).encode("utf-8")).hexdigest()[:20]


def _read_uploaded_file(file_data: bytes, name: str, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(BytesIO(file_data), sheet_name=sheet_name or 0)
    if suffix in {".csv", ".txt"}:
        try:
            return pd.read_csv(BytesIO(file_data))
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(file_data), encoding="latin-1")
    raise ValueError("Unsupported file type. Please upload .xlsx, .xls, .xlsm, or .csv.")


def _show_severity_counts(issues: pd.DataFrame) -> None:
    if issues.empty:
        st.success("No automated checks raised a warning. This does not guarantee that the data are leakage-free or clinically valid.")
        return
    cols = st.columns(4)
    for col, severity in zip(cols, ["CRITICAL", "HIGH", "MEDIUM", "LOW"]):
        count = int((issues["severity"] == severity).sum())
        col.metric(severity.title(), count)


def _download_button(label: str, data: bytes, filename: str, mime: str, key: str) -> None:
    st.download_button(label, data=data, file_name=filename, mime=mime, key=key)


uploaded = st.file_uploader("Step 1 — Upload your dataset", type=["xlsx", "xls", "xlsm", "csv", "txt"])
if uploaded is None:
    st.info("Start with an Excel workbook or CSV. A synthetic example is included in the GitHub repository under `sample_data/`.")
    st.markdown("**Important:** identify one row per independent prediction unit where possible. If multiple rows belong to the same person/site/family, use a group-based split rather than a random row split.")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    suffix = Path(uploaded.name).suffix.lower()
    sheet_name = None
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        xls = pd.ExcelFile(BytesIO(file_bytes))
        if not xls.sheet_names:
            st.error("The Excel workbook contains no worksheets.")
            st.stop()
        sheet_name = st.selectbox("Worksheet", options=xls.sheet_names, index=0)
    df = _read_uploaded_file(file_bytes, uploaded.name, sheet_name)
    df.columns = [str(column).strip() for column in df.columns]
except Exception as exc:
    st.error(f"Could not read the uploaded file: {type(exc).__name__}: {exc}")
    st.stop()

file_signature = _file_fingerprint(file_bytes, sheet_name)
st.session_state["current_df"] = df
st.session_state["current_file_signature"] = file_signature

with st.expander("Preview and import details", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    st.dataframe(df.head(20), use_container_width=True)

if df.empty or df.shape[1] == 0:
    st.error("The uploaded data contain no rows or no columns. Correct the worksheet and upload it again.")
    st.stop()
if not df.columns.is_unique:
    st.error("Duplicate column names were detected by pandas. Rename them so every column has a unique name before continuing.")
    st.stop()

st.divider()
st.header("Step 2 — Define the prediction task and audit the data")
target = st.selectbox("Outcome / target column", options=list(df.columns), index=0, help="This is the variable the model will predict. Choose an outcome known only after the predictor values would be available in real use.")
suggested_task = suggest_task_type(df[target])
task_options = ["classification", "regression"]
default_task_index = task_options.index(suggested_task)
task = st.radio("Problem type (confirm the suggestion)", options=task_options, index=default_task_index, horizontal=True)
if suggested_task != task:
    st.warning(f"The automatic hint was **{suggested_task}**, but you selected **{task}**. Confirm that this matches the scientific meaning of the outcome; numeric labels can still represent classes.")
positive_class = None
if task == "classification":
    target_values = df[target].dropna().unique().tolist()
    st.caption(f"Observed target classes: {df[target].nunique(dropna=True)}. Confirm the event/positive label for binary outcomes; this affects the interpretation of probability and class-wise metrics.")
    if len(target_values) == 2:
        positive_class = st.selectbox(
            "Positive / event class (required for binary classification)",
            options=[None] + target_values,
            index=0,
            format_func=lambda value: "— Select the event class —" if value is None else str(value),
            help="Choose the class representing the event of interest. The app codes this class as 1 internally for binary metrics, then shows the original label in the report.",
            key=f"positive_class_{file_signature}_{target}",
        )
        if positive_class is None:
            st.warning("Select the event/positive class before preparing a binary classification dataset.")

current_audit_signature = f"{file_signature}:{target}:{task}"
if st.button("Run data-quality audit", type="primary", key="run_audit"):
    issues, summary = audit_dataframe(df, target=target)
    st.session_state["audit_result"] = {"signature": current_audit_signature, "issues": issues, "summary": summary}
    # A target or file change invalidates all downstream results.
    for key in ["prepared_result", "benchmark_result", "final_result", "export_xlsx", "export_model"]:
        st.session_state.pop(key, None)

audit_result = st.session_state.get("audit_result")
audit_is_current = bool(audit_result and audit_result.get("signature") == current_audit_signature)
if audit_is_current:
    issues = audit_result["issues"]
    summary = audit_result["summary"]
    _show_severity_counts(issues)
    if not issues.empty:
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        display_issues = issues.assign(_order=issues["severity"].map(severity_order)).sort_values(["_order", "check"]).drop(columns="_order")
        st.dataframe(display_issues, use_container_width=True, hide_index=True)
        st.caption("Warnings are prompts for human review. The program does not know the data-collection protocol and will not silently delete variables just because their names look suspicious.")
    else:
        st.success("No automatic issues detected. Review the study design and prediction-time availability anyway.")
    if task == "classification":
        target_counts = df[target].value_counts(dropna=False).rename_axis("class").reset_index(name="count")
        target_counts["percentage_of_all_rows"] = target_counts["count"] / max(1, len(df)) * 100
        with st.expander("Target distribution", expanded=True):
            st.dataframe(target_counts, use_container_width=True, hide_index=True)
    if issues["severity"].eq("CRITICAL").any():
        st.error("Resolve critical issues before preparing a model. A target with only one observed class, for example, cannot support supervised classification.")
else:
    st.info("Run the audit to unlock preparation and modelling. Changing the file, target, or task requires a fresh audit.")

if audit_is_current and not issues["severity"].eq("CRITICAL").any():
    st.divider()
    st.header("Step 3 — Prepare features and choose a validation design")
    id_suggestions = list(summary.get("identifier_candidates", []))
    leak_suggestions = list(summary.get("possible_leakage_candidates", []))
    preselected_exclusions = [c for c in dict.fromkeys(id_suggestions + leak_suggestions) if c in df.columns and c != target]
    feature_candidates = [c for c in df.columns if c != target]
    excluded_columns = st.multiselect(
        "Columns to exclude from predictors",
        options=feature_candidates,
        default=preselected_exclusions,
        help="Identifiers and possible post-outcome leakage candidates are preselected as a safety prompt. You may change this after reviewing each column's meaning.",
        key=f"exclude_{file_signature}_{target}",
    )
    date_suggestions = [c for c in detect_date_columns(df) if c != target and c not in excluded_columns]
    date_columns = st.multiselect(
        "Date columns to expand into year/month/day/day-of-week features",
        options=[c for c in feature_candidates if c not in excluded_columns],
        default=date_suggestions,
        help="Only select dates that would be available at the intended prediction time. Raw selected date columns are replaced by simple calendar features.",
        key=f"dates_{file_signature}_{target}",
    )
    split_strategy = st.selectbox(
        "Holdout and cross-validation strategy",
        options=["Random (stratified for classification)", "Group-based (same group never crosses splits)", "Time-based (train on earlier, test on later rows)"],
        index=0,
        help="Use group-based splitting for repeated people/sites/households. Use time-based splitting when the deployment goal is prediction on future observations.",
        key=f"split_{file_signature}_{target}",
    )
    group_col = None
    time_col = None
    if split_strategy.startswith("Group"):
        group_options = [c for c in df.columns if c != target]
        group_col = st.selectbox("Group identifier column", options=group_options, index=0, key=f"group_{file_signature}_{target}")
        if group_col not in excluded_columns:
            st.warning("The group identifier must be excluded from the predictor features. It will be excluded automatically when preparing data.")
    if split_strategy.startswith("Time"):
        time_options = [c for c in df.columns if c != target]
        time_col = st.selectbox("Time/order column", options=time_options, index=0, key=f"time_{file_signature}_{target}")
        if time_col not in excluded_columns:
            st.warning("The time/order column will be excluded as a raw predictor. Select it as a date feature separately only when calendar time is genuinely available at prediction time.")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_fraction = st.slider("Final test-set fraction", min_value=0.10, max_value=0.40, value=0.20, step=0.05, key=f"test_frac_{file_signature}_{target}")
    with col2:
        cv_folds = st.slider("Cross-validation folds", min_value=3, max_value=10, value=5, step=1, key=f"cv_folds_{file_signature}_{target}")
    with col3:
        feature_selection = st.selectbox("Feature selection", options=["None", "ANOVA F-score", "Model-based (Extra Trees)"], index=0, key=f"fs_{file_signature}_{target}")
    k_features = st.slider("Top transformed features to retain (when feature selection is enabled)", min_value=5, max_value=100, value=20, step=5, key=f"k_features_{file_signature}_{target}")
    n_classes = int(df[target].dropna().nunique()) if task == "classification" else 0
    metrics = metric_catalog(task, n_classes=n_classes)
    default_metric = "roc_auc" if task == "classification" and n_classes == 2 else ("roc_auc_ovr" if task == "classification" else "rmse")
    if default_metric not in metrics:
        default_metric = "f1_macro" if task == "classification" else "rmse"
    metric_options = list(metrics.keys())
    primary_metric = st.selectbox(
        "Primary metric for ranking candidate models",
        options=metric_options,
        index=metric_options.index(default_metric),
        format_func=lambda key: metrics[key]["label"],
        help="Choose this based on the scientific use case and error costs, not on whichever metric gives the largest number.",
        key=f"metric_{file_signature}_{target}",
    )
    with st.expander("Optional hyperparameter tuning", expanded=False):
        tune_enabled = st.checkbox("Tune the top 3 models after initial screening", value=False, key=f"tune_{file_signature}_{target}")
        tuning_iterations = st.slider("Randomized-search iterations per finalist", min_value=3, max_value=20, value=5, step=1, disabled=not tune_enabled, key=f"tune_iter_{file_signature}_{target}")
        st.caption("Tuning can substantially increase run time. Model screening and tuning use training data only; the holdout set is reserved for the final evaluation.")

    config = {
        "file_signature": file_signature,
        "target": target,
        "task": task,
        "positive_class": positive_class,
        "excluded_columns": tuple(sorted(set(excluded_columns + ([group_col] if group_col else []) + ([time_col] if time_col else [])))),
        "date_columns": tuple(sorted(date_columns)),
        "split_strategy": split_strategy,
        "group_col": group_col,
        "time_col": time_col,
        "test_fraction": float(test_fraction),
        "cv_folds": int(cv_folds),
        "feature_selection": feature_selection,
        "k_features": int(k_features),
        "primary_metric": primary_metric,
        "tune_enabled": bool(tune_enabled),
        "tuning_iterations": int(tuning_iterations),
    }
    signature = hashlib.sha256(repr(config).encode("utf-8")).hexdigest()[:20]

    if st.button("Prepare data and create the train/test split", type="primary", key="prepare_data"):
        try:
            if not df.columns.is_unique:
                raise ValueError("Column names must be unique.")
            if task == "classification" and df[target].dropna().nunique() < 2:
                raise ValueError("The target has fewer than two classes.")
            # Explicitly convert target to clean, non-null supervised observations in one place.
            X, y, prep_metadata = prepare_features(df, target, excluded_columns=config["excluded_columns"], date_columns=date_columns)
            source_row_numbers = prep_metadata.pop("source_row_numbers", list(range(2, len(y) + 2)))
            class_label_mapping = {}
            if task == "classification" and y.nunique() == 2:
                if positive_class is None:
                    raise ValueError("Select the positive/event class above before preparing binary classification data.")
                original_classes = y.dropna().unique().tolist()
                if positive_class not in original_classes:
                    raise ValueError("The selected positive class is no longer present in the labelled target rows. Re-select the target class.")
                negative_class = next(value for value in original_classes if value != positive_class)
                y = y.map({negative_class: 0, positive_class: 1}).astype(int)
                class_label_mapping = {0: negative_class, 1: positive_class}
                prep_metadata["binary_target_encoding"] = {"0 (negative)": str(negative_class), "1 (positive/event)": str(positive_class)}
            elif task == "regression":
                numeric_target = pd.to_numeric(y, errors="coerce")
                invalid_target_count = int(numeric_target.isna().sum())
                if invalid_target_count:
                    raise ValueError(f"Regression requires a numeric target. {invalid_target_count} labelled value(s) could not be converted; clean the target or choose classification if it represents classes.")
                y = numeric_target.astype(float)
            valid_target_mask = df[target].notna()
            groups = df.loc[valid_target_mask, group_col].reset_index(drop=True) if group_col else None
            times = df.loc[valid_target_mask, time_col].reset_index(drop=True) if time_col else None
            train_idx, test_idx = make_holdout_split(
                X, y, task, strategy=split_strategy, test_size=test_fraction,
                groups=groups, times=times,
            )
            if task == "classification":
                train_classes = y.iloc[train_idx].nunique()
                test_classes = y.iloc[test_idx].nunique()
                if train_classes < 2:
                    raise ValueError("The training partition contains fewer than two classes. Change the split strategy/fraction or collect more data.")
                if test_classes < 2:
                    st.warning("The holdout contains fewer than two target classes. AUC may be undefined; interpret final metrics cautiously.")
            st.session_state["prepared_result"] = {
                "signature": signature,
                "config": config,
                "X": X,
                "y": y,
                "groups": groups,
                "times": times,
                "source_row_numbers": source_row_numbers,
                "class_label_mapping": class_label_mapping,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "metadata": prep_metadata,
                "audit_issues": issues,
                "audit_summary": summary,
            }
            for key in ["benchmark_result", "final_result", "export_xlsx", "export_model"]:
                st.session_state.pop(key, None)
        except Exception as exc:
            st.error(f"Preparation could not continue: {type(exc).__name__}: {exc}")

    prepared = st.session_state.get("prepared_result")
    prepared_is_current = bool(prepared and prepared.get("signature") == signature)
    if prepared_is_current:
        st.success(f"Prepared {len(prepared['y']):,} labelled rows with {prepared['X'].shape[1]:,} predictor columns.")
        st.json(prepared["metadata"])
        st.write(f"Training rows: **{len(prepared['train_idx']):,}** · Held-out test rows: **{len(prepared['test_idx']):,}**")
        if prepared["X"].shape[0] < 50:
            st.warning("This is a small dataset for broad automated model comparison. Performance estimates may be highly uncertain, even with cross-validation.")
        st.markdown("**Check before proceeding:** ensure there is no duplicate-person/site leakage, no predictor recorded after the outcome, and no preprocessing decision informed by the holdout outcomes.")

        st.divider()
        st.header("Step 4 — Compare candidate models using training-only cross-validation")
        if st.button("Benchmark models", type="primary", key="benchmark_models"):
            try:
                with st.spinner("Fitting candidate models using training data only…"):
                    benchmark = benchmark_models(
                        prepared["X"], prepared["y"], task,
                        prepared["train_idx"], prepared["test_idx"], split_strategy,
                        groups=prepared["groups"], times=prepared["times"],
                        primary_metric=primary_metric,
                        feature_selection=feature_selection,
                        k_features=k_features,
                        requested_cv_splits=cv_folds,
                        tune_top_n=3 if tune_enabled else 0,
                        n_iter=tuning_iterations,
                    )
                st.session_state["benchmark_result"] = {"signature": signature, "result": benchmark}
                st.session_state.pop("final_result", None)
                st.session_state.pop("export_xlsx", None)
                st.session_state.pop("export_model", None)
            except Exception as exc:
                st.error(f"Benchmarking could not complete: {type(exc).__name__}: {exc}")
        benchmark_state = st.session_state.get("benchmark_result")
        benchmark_is_current = bool(benchmark_state and benchmark_state.get("signature") == signature)
        if benchmark_is_current:
            benchmark = benchmark_state["result"]
            st.success(f"Selected by training cross-validation: **{benchmark['best_model_name']}** (primary metric: {metrics[primary_metric]['label']}).")
            if benchmark["dense_models_skipped"]:
                st.warning(f"Dense-only algorithms were skipped because the estimated expanded matrix contains {benchmark['dense_cells_estimate']:,} cells, above the safety limit. Sparse-compatible models were still benchmarked.")
            display_cv = benchmark["cv_table"].copy()
            # The raw selection scorer is negative for loss metrics such as RMSE/MAE. Display native-scale metrics instead.
            display_cv = display_cv.drop(columns=[c for c in ["cv_selection_score_mean", "cv_selection_score_sd"] if c in display_cv.columns])
            lead_columns = [c for c in ["selected_for_final_test", "model", "selection_stage", "status", "cv_folds", "cv_primary_metric", "cv_primary_mean", "cv_primary_sd", "tuning_finalist", "tuned"] if c in display_cv.columns]
            display_cv = display_cv[lead_columns + [c for c in display_cv.columns if c not in lead_columns]]
            st.dataframe(display_cv, use_container_width=True, hide_index=True)
            st.caption("Primary metric values are shown on their native scale. For loss metrics such as RMSE/MAE, lower is better; for accuracy/AUC/F1/R², higher is better. When tuning is enabled, only the top three screening finalists are tuned and compared for the final model choice; other rows are the initial screen only.")
            if benchmark["tuning_notes"]:
                with st.expander("Hyperparameter tuning notes"):
                    for note in benchmark["tuning_notes"]:
                        st.write(f"- {note}")
            st.caption("CV scores are training-only estimates. Candidate selection is not based on the held-out test outcomes.")

            st.divider()
            st.header("Step 5 — Final held-out evaluation")
            st.warning("Run this only after the target, exclusions, validation design, primary metric, and feature-selection choices are finalized. Repeatedly changing the workflow based on test results turns the test set into validation data.")
            if st.button("Evaluate selected model on the held-out test set", type="primary", key="evaluate_final"):
                try:
                    with st.spinner("Fitting the selected pipeline on the full training partition and evaluating the holdout…"):
                        final_eval = evaluate_final_model(
                            benchmark, prepared["X"], prepared["y"], task,
                            primary_metric=primary_metric,
                            row_ids=prepared.get("source_row_numbers"),
                            class_label_mapping=prepared.get("class_label_mapping"),
                        )
                    st.session_state["final_result"] = {"signature": signature, "result": final_eval}
                except Exception as exc:
                    st.error(f"Final evaluation could not complete: {type(exc).__name__}: {exc}")
            final_state = st.session_state.get("final_result")
            final_is_current = bool(final_state and final_state.get("signature") == signature)
            if final_is_current:
                final_eval = final_state["result"]
                st.subheader(f"Held-out performance — {benchmark['best_model_name']}")
                metric_cols = st.columns(min(4, max(1, len(final_eval["metrics"]))))
                main_metrics = [(k, v) for k, v in final_eval["metrics"].items() if isinstance(v, (float, int, np.floating, np.integer)) and np.isfinite(v)]
                for i, (key, value) in enumerate(main_metrics[:len(metric_cols)]):
                    metric_cols[i].metric(key.replace("_", " ").title(), f"{float(value):.4f}")
                metric_table = pd.DataFrame([{"metric": k, "value": v} for k, v in final_eval["metrics"].items()])
                st.dataframe(metric_table, use_container_width=True, hide_index=True)
                if final_eval["metrics"].get("auc_note"):
                    st.warning(final_eval["metrics"]["auc_note"])
                if not final_eval["confusion_matrix"].empty:
                    st.subheader("Confusion matrix")
                    st.dataframe(final_eval["confusion_matrix"], use_container_width=True, hide_index=True)
                if not final_eval["classification_report"].empty:
                    with st.expander("Class-wise precision, recall, and F1"):
                        st.dataframe(final_eval["classification_report"], use_container_width=True, hide_index=True)
                with st.expander("Holdout predictions", expanded=False):
                    st.dataframe(final_eval["predictions"].head(200), use_container_width=True, hide_index=True)
                with st.expander("Permutation feature importance", expanded=True):
                    st.caption("This model-specific, permutation-based estimate uses the holdout features and labels for interpretation; do not use it to reselect the model and then report the same holdout as untouched.")
                    st.dataframe(final_eval["permutation_importance"].head(50), use_container_width=True, hide_index=True)

                if st.button("Build downloadable report and model", key="build_exports"):
                    export_config = dict(config)
                    export_config["selected_model"] = benchmark["best_model_name"]
                    export_config["primary_metric"] = primary_metric
                    export_config["input_file_name"] = uploaded.name
                    export_config["worksheet"] = sheet_name or "CSV/text input"
                    export_config["python_version"] = sys.version.split()[0]
                    export_config["pandas_version"] = pd.__version__
                    export_config["numpy_version"] = np.__version__
                    export_config["scikit_learn_version"] = sklearn.__version__
                    export_config["random_state"] = 42
                    report_bytes = excel_report_bytes(
                        prepared["audit_issues"], prepared["audit_summary"], prepared["metadata"],
                        benchmark["cv_table"], final_eval, export_config,
                    )
                    predictor = TabularPredictor(
                        estimator=final_eval["model"],
                        target_column=target,
                        excluded_columns=list(config["excluded_columns"]),
                        date_columns=list(config["date_columns"]),
                        numeric_columns=prepared["X"].select_dtypes(include=[np.number]).columns.astype(str).tolist(),
                        expected_columns=prepared["X"].columns.astype(str).tolist(),
                        class_label_mapping=prepared.get("class_label_mapping"),
                    )
                    fitted_bytes = model_bytes(predictor)
                    st.session_state["export_xlsx"] = {"signature": signature, "bytes": report_bytes}
                    st.session_state["export_model"] = {"signature": signature, "bytes": fitted_bytes}
                xlsx_state = st.session_state.get("export_xlsx")
                model_state = st.session_state.get("export_model")
                if xlsx_state and xlsx_state.get("signature") == signature:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        _download_button("Download Excel report", xlsx_state["bytes"], "ml_workbench_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "download_xlsx")
                    if model_state and model_state.get("signature") == signature:
                        with col_b:
                            _download_button("Download fitted model (.joblib)", model_state["bytes"], "selected_pipeline.joblib", "application/octet-stream", "download_model")
                    st.caption("A joblib artifact can execute code when loaded: load only model files you created or otherwise trust. The exported predictor accepts raw rows, performs the configured date/numeric preparation, ignores excluded columns, and returns original binary labels. Missing expected predictors are handled by the fitted pipeline's imputation steps.")
    else:
        st.info("Prepare the data and confirm the proposed train/test split to unlock model benchmarking.")

st.divider()
st.caption("Research use only. This application is a starting point for tabular predictive modelling, not a substitute for a prespecified statistical analysis plan, external validation, calibration assessment, fairness review, or domain-expert validation. For clustered, longitudinal, censored, time-to-event, image, text-heavy, or very high-dimensional data, the generic workflow may need to be adapted.")
