"""Export human-readable Excel reports and serialized model pipelines."""
from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def excel_report_bytes(
    audit_issues: pd.DataFrame,
    audit_summary: dict[str, Any],
    preparation: dict[str, Any],
    cv_table: pd.DataFrame,
    final_evaluation: dict[str, Any] | None,
    configuration: dict[str, Any],
) -> bytes:
    """Create a multi-sheet .xlsx report in memory."""
    summary_rows = [{"item": k, "value": str(v)} for k, v in audit_summary.items()]
    summary_rows.extend({"item": f"preparation.{k}", "value": str(v)} for k, v in preparation.items())
    summary_rows.extend({"item": f"configuration.{k}", "value": str(v)} for k, v in configuration.items())
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        (audit_issues if not audit_issues.empty else pd.DataFrame(columns=["severity", "check", "finding", "recommended_action", "affected_columns"])).to_excel(writer, sheet_name="Data audit", index=False)
        cv_table.to_excel(writer, sheet_name="CV model comparison", index=False)
        if final_evaluation is not None:
            pd.DataFrame([{"metric": k, "value": v} for k, v in final_evaluation["metrics"].items()]).to_excel(writer, sheet_name="Holdout metrics", index=False)
            final_evaluation["predictions"].to_excel(writer, sheet_name="Holdout predictions", index=False)
            if not final_evaluation["classification_report"].empty:
                final_evaluation["classification_report"].to_excel(writer, sheet_name="Classification report", index=False)
            if not final_evaluation["confusion_matrix"].empty:
                final_evaluation["confusion_matrix"].to_excel(writer, sheet_name="Confusion matrix", index=False)
            if not final_evaluation["permutation_importance"].empty:
                final_evaluation["permutation_importance"].to_excel(writer, sheet_name="Feature importance", index=False)
    return buffer.getvalue()


def model_bytes(model: Any) -> bytes:
    """Serialize a fitted sklearn Pipeline. Load only trusted artifacts."""
    import joblib
    buffer = BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.getvalue()
