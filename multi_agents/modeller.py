from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget, RevertToUserTarget
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Annotated, Literal, Any
import os
import json


# ==============================
# Output schema
# ==============================
class ModelerOutput(BaseModel):
    best_model_name: str = Field(..., description="Name of the best estimator selected by FLAML.")
    best_metric_name: str = Field(..., description="Primary metric used for selection.")
    best_metric_value: float = Field(..., description="Score on the validation set for the primary metric.")
    model_artifact_path: str = Field(..., description="Path to the persisted best model artifact.")
    cv_summary_path: Optional[str] = Field(
        None, description="Path to saved JSON/CSV with training summary (if produced)."
    )


# ==============================
# Small helpers (paths + metrics)
# ==============================
def _ensure_paths(paths: Dict[str, str], required: List[str]) -> Optional[str]:
    """Return None if all good, else a short error string."""
    for k in required:
        if k not in paths or not paths[k] or not os.path.exists(paths[k]):
            return f"Missing or invalid path for '{k}'."
    return None


def _default_metric(task_type: str) -> str:
    # FLAML metric names
    return "r2" if task_type == "Regression" else "accuracy"


def _metric_direction(metric: str) -> Literal["minimize", "maximize"]:
    # FLAML knows direction internally; we keep for reporting consistency if needed
    return "maximize" if metric in {"r2", "accuracy", "roc_auc"} else "minimize"


# ==============================
# Tools
# ==============================
def request_clarification(
    clarification_question: Annotated[str, "One targeted question to unblock modeling (e.g., 'Which primary metric?')"],
) -> ReplyResult:
    return ReplyResult(
        message=f"Further clarification is required:\n- {clarification_question}",
        target=RevertToUserTarget(),
    )


def train_with_flaml(
    data_splits_paths: Annotated[
        Dict[str, str],
        "Paths to preprocessed splits. Expect keys: X_train, y_train, X_val, y_val (and optionally X_test, y_test)."
    ],
    task_type: Annotated[Literal["Regression", "Classification"], "Supported tasks"] = "Regression",
    primary_metric: Optional[str] = None,   # FLAML metric name: e.g., r2, mse, mae, accuracy, roc_auc
    time_budget_s: int = 60,
    estimator_list: Optional[List[str]] = None,  # e.g., ["lgbm","xgboost","rf","extra_tree","lrl1","catboost"]
    output_dir: str = "./artifacts",
    context_variables: ContextVariables = None,
) -> ReplyResult:
    """
    Train with FLAML AutoML on the training split, evaluate on validation, and persist the best model.
    """
    # Validate basic inputs
    need = ["X_train", "y_train", "X_val", "y_val"]
    msg = _ensure_paths(data_splits_paths, need)
    if msg:
        return ReplyResult(
            message=f"Cannot start modeling: {msg}",
            target=AgentNameTarget("Modeler"),
            context_variables=context_variables,
        )

    # Lazy imports to keep agent light
    import pandas as pd
    from flaml import AutoML
    import joblib
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, roc_auc_score, f1_score
    )

    os.makedirs(output_dir, exist_ok=True)
    primary_metric = primary_metric or _default_metric(task_type)

    # Load data
    X_train = pd.read_csv(data_splits_paths["X_train"])
    y_train = pd.read_csv(data_splits_paths["y_train"]).iloc[:, 0]
    X_val   = pd.read_csv(data_splits_paths["X_val"])
    y_val   = pd.read_csv(data_splits_paths["y_val"]).iloc[:, 0]

    # Configure FLAML
    automl = AutoML()
    flaml_settings = {
        "time_budget": time_budget_s,
        "task": "regression" if task_type == "Regression" else "classification",
        "metric": primary_metric,        # e.g., r2 / accuracy / roc_auc / mse / mae
        "log_file_name": os.path.join(output_dir, "flaml_training.log"),
        "estimator_list": estimator_list,  # None lets FLAML choose
        "verbose": 0,
        # You may also pass CV here, but we assume DataEngineer already produced a val split.
    }

    # Fit
    automl.fit(X_train=X_train, y_train=y_train, eval_set=[(X_val, y_val)], **flaml_settings)

    # Evaluate on validation with scikit metrics for clarity
    y_val_pred = automl.predict(X_val)
    if task_type == "Regression":
        metric_map = {
            "r2": r2_score,
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        }
    else:
        # For classification, try to compute the requested metric; fallback to accuracy
        metric_map = {
            "accuracy": accuracy_score,
            "roc_auc": roc_auc_score,  # requires probabilities & binary/multiclass handling
            "f1": f1_score,            # defaults to 'binary' averaging if labels are binary
        }

    # Compute primary metric as best we can
    best_metric_value: float
    try:
        if task_type == "Classification" and primary_metric in {"roc_auc"}:
            # Need predicted probabilities
            if hasattr(automl.model, "predict_proba"):
                proba = automl.predict_proba(X_val)
                # Handle binary vs multiclass
                if proba.shape[1] == 2:
                    best_metric_value = roc_auc_score(y_val, proba[:, 1])
                else:
                    # macro-average AUC if multiclass
                    best_metric_value = roc_auc_score(y_val, proba, multi_class="ovr", average="macro")
            else:
                # Fallback to accuracy if proba is not available
                best_metric_value = accuracy_score(y_val, y_val_pred)
                primary_metric = "accuracy"
        else:
            scorer = metric_map.get(primary_metric)
            if scorer is None:
                # Reasonable fallback per task
                best_metric_value = r2_score(y_val, y_val_pred) if task_type == "Regression" else accuracy_score(y_val, y_val_pred)
                primary_metric = "r2" if task_type == "Regression" else "accuracy"
            else:
                # For f1 on multiclass, switch to macro average
                if scorer is f1_score:
                    # detect multiclass
                    avg = "binary" if len(set(y_val)) == 2 else "macro"
                    best_metric_value = f1_score(y_val, y_val_pred, average=avg)
                else:
                    best_metric_value = float(scorer(y_val, y_val_pred))
    except Exception:
        # Absolute fallback to accuracy/r2
        if task_type == "Regression":
            best_metric_value = float(r2_score(y_val, y_val_pred))
            primary_metric = "r2"
        else:
            best_metric_value = float(accuracy_score(y_val, y_val_pred))
            primary_metric = "accuracy"

    # Persist best model
    best_model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump(automl, best_model_path)

    # Try to provide a minimal “candidates” list from flaml's internal history if available
    candidates: List[Dict[str, Any]] = []
    try:
        # automl.model_history is not a public attr; safer: use best_estimator & config history if present
        if hasattr(automl, "best_estimator") and hasattr(automl, "best_config"):
            candidates.append({
                "estimator": automl.best_estimator,
                "config": automl.best_config,
                "val_metric": best_metric_value,
            })
    except Exception:
        pass

    # Optionally include test evaluation if paths exist
    test_metric_record = {}
    if all(k in data_splits_paths for k in ("X_test", "y_test")) and \
       os.path.exists(data_splits_paths["X_test"]) and os.path.exists(data_splits_paths["y_test"]):
        X_test = pd.read_csv(data_splits_paths["X_test"])
        y_test = pd.read_csv(data_splits_paths["y_test"]).iloc[:, 0]
        y_test_pred = automl.predict(X_test)
        try:
            if task_type == "Regression":
                test_metric_record["r2"] = float(r2_score(y_test, y_test_pred))
                test_metric_record["mae"] = float(mean_absolute_error(y_test, y_test_pred))
                test_metric_record["mse"] = float(mean_squared_error(y_test, y_test_pred))
            else:
                test_metric_record["accuracy"] = float(accuracy_score(y_test, y_test_pred))
                if len(set(y_test)) == 2 and hasattr(automl.model, "predict_proba"):
                    proba_t = automl.predict_proba(X_test)
                    test_metric_record["roc_auc"] = float(roc_auc_score(y_test, proba_t[:, 1]))
        except Exception:
            pass

    # Save a small JSON summary
    summary = {
        "task_type": task_type,
        "primary_metric": primary_metric,
        "val_primary_metric": best_metric_value,
        "best_estimator": getattr(automl, "best_estimator", "unknown"),
        "test_metrics": test_metric_record,
        "time_budget_s": time_budget_s,
        "estimator_list": estimator_list,
    }
    cv_summary_path = os.path.join(output_dir, "model_summary.json")
    with open(cv_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Build output
    out = ModelerOutput(
        best_model_name=getattr(automl, "best_estimator", "unknown"),
        best_metric_name=primary_metric,
        best_metric_value=float(best_metric_value),
        model_artifact_path=best_model_path,
        candidates=candidates,
        cv_summary_path=cv_summary_path,
    )

    # Stash to context & handoff to Evaluator
    if context_variables is None:
        context_variables = {}
    context_variables.update(out.model_dump())

    return ReplyResult(
        message="Model training complete via FLAML. Best model persisted.",
        target=AgentNameTarget("Evaluator"),
        context_variables=context_variables,
    )


def complete_modeler_task(
    results: ModelerOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    """
    If you call the modeler from somewhere else and already have results, finalize & pass to Evaluator.
    """
    context_variables.update(results.model_dump())
    return ReplyResult(
        message="Modeling complete. Passing artifacts and metrics to Evaluator.",
        target=AgentNameTarget("Evaluator"),
        context_variables=context_variables,
    )


def execute_modeling_plan(
    plan: Annotated[str, "One small modeling step to delegate to the Coder (optional)"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Optional: tiny, focused delegation to a Coder agent (e.g., 'plot feature importance for best model').
    """
    context_variables["current_agent"] = "Modeler"
    return ReplyResult(
        message=f"Please write Python code to execute this modeling step:\n{plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )


# ==============================
# Agent definition
# ==============================
class Modeler(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="Modeler",
            llm_config=LLMConfig(
                api_type="openai",
                model="gpt-4o-mini",
                response_format=ModelerOutput,   # used when calling complete_modeler_task directly
                parallel_tool_calls=False,
            ),
            system_message="""
You are the Modeler.
- Use Microsoft FLAML AutoML to train and select the best model without manual hyperparameter tuning.

Workflow:
1) Expect preprocessed data splits from DataEngineer with keys: X_train, y_train, X_val, y_val (and optionally X_test, y_test).
2) If the task_type is not 'Regression' or 'Classification', or a required split path is missing, ask ONE targeted question via `request_clarification`.
3) Call `train_with_flaml` with a sensible time budget (default 60s) and the primary metric (defaults: r2 for Regression, accuracy for Classification).
4) Persist the best model and a short JSON summary; then hand off to Evaluator.

Rules:
- Do NOT perform raw data cleaning; rely on DataEngineer outputs.
- Keep interactions minimal and deterministic; no full data prints.
- If metric is incompatible (e.g., roc_auc without predict_proba), fallback to a safe metric like accuracy.
            """.strip(),
            functions=[
                request_clarification,
                train_with_flaml,
                complete_modeler_task,
                execute_modeling_plan,
            ],
        )
