from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field

class DataScientistStep(BaseModel):
    step_description: str = Field(
        ...,
        description="Modeling step to perform.",
        examples=[
            "Load cleaned datasets produced by the Data Engineer (e.g., ./data/train_cleaned.csv, ./data/val_cleaned.csv, ./data/test_cleaned.csv)",
            "Define the processing pipeline: categorical → OneHotEncoder → StandardScaler; numerical → StandardScaler; datetime → extract [day, month] → cosine transform → StandardScaler",
            "Build per-model training pipelines that combine the processing pipeline with each estimator",
            "Do NOT fit the preprocessing pipeline alone; ONLY fit the combined (processing + model) pipeline",
            "Fit every training pipeline on the loaded train_cleaned split only (no leakage), then validate on val_cleaned and, if available, test on test_cleaned",
            "Hyperparameter tuning with proper CV and selection of the best model by the primary metric",
            "Persist the best fitted end-to-end pipeline (processing + model) to a specified path for deployment"
            "Save the best model and its parameters for deployment at sepecific path (e.g., ./artifacts/best_model_{problem_type}.pkl)"
        ]
    )
    instruction: str = Field(
        ...,
        description=(
            "Implement the workflow in this exact order: "
            "(1) Load the cleaned splits produced by the Data Engineer (train_cleaned, val_cleaned, test_cleaned if available). "
            "(2) Build ONE processing pipeline via ColumnTransformer: categorical → OneHotEncoder(handle_unknown='ignore') → StandardScaler(with_mean=False); numerical → StandardScaler; datetime → extract [day, month] → cosine transform → StandardScaler. "
            "(3) For each candidate estimator, MUST create ONE sklearn Pipeline: Pipeline([('preprocess', processing_pipeline), ('est', estimator)]). "
            "(4) NEVER call fit()/transform() on the processing pipeline by itself; call fit() ONLY on the combined pipeline using the train_cleaned split. "
            "(5) Choose the correct cross-validation (StratifiedKFold/KFold/TimeSeriesSplit) and scoring for {problem_type}; when tuning, pass parameters with the 'est__' prefix. "
            "(6) ALWAYS fit on (X_train_cleaned, y_train_cleaned) only (no validation/test leakage); then evaluate on val_cleaned and, if available, test_cleaned. "
            "(7) Persist the best fitted pipeline (processing + estimator) to the specified artifact path (e.g., ./artifacts/best_model_{problem_type}.joblib) and record key metrics."

        ),
        examples=[
            "Load ./data/train_cleaned.csv, ./data/val_cleaned.csv, ./data/test_cleaned.csv; set X_* and y_* using the 'target' column.",
            "Create a processing pipeline: categorical → OneHotEncoder → StandardScaler(with_mean=False if sparse); numerical → StandardScaler; "
            "datetime → extract day & month → cosine transform (encode cyclicality) → StandardScaler.",
            "Assemble per-model pipelines: Pipeline([('preprocess', processing_pipeline), ('est', estimator)]) for each candidate estimator.",
            "Do NOT fit the preprocessing pipeline in isolation. Call fit() ONLY on the full pipeline using (X_train_cleaned, y_train_cleaned); "
            "evaluate on (X_val_cleaned, y_val_cleaned), and report test metrics if (X_test_cleaned, y_test_cleaned) exist.",
            "Use GridSearchCV/RandomizedSearchCV with the correct splitter (StratifiedKFold/KFold/TimeSeriesSplit) and an appropriate scoring metric.",
            "Select the best pipeline by the primary metric and persist it (processing + model) to ./artifacts/best_model_{problem_type}.joblib, with key metrics."
            
            """
            # Full example: end-to-end pipeline (processing + model) + Grid Search

            import pandas as pd
            import numpy as np
            from pathlib import Path
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            from sklearn.model_selection import GridSearchCV, KFold
            import joblib

            # 1) Load cleaned splits (produced by Data Engineer)
            train = pd.read_csv("./data/train_cleaned.csv")
            val   = pd.read_csv("./data/val_cleaned.csv")
            # (Optional) test = pd.read_csv("./data/test_cleaned.csv")

            TARGET = "target"
            X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
            X_val,   y_val   = val.drop(columns=[TARGET]),   val[TARGET]

            # 2) Identify column types (example heuristics)
            cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
            num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

            # Try to coerce likely datetime columns (if any)
            for c in X_train.columns:
                if c.lower().endswith(("date", "_dt")) or ("date" in c.lower()):
                    X_train[c] = pd.to_datetime(X_train[c], errors="ignore")
                    X_val[c]   = pd.to_datetime(X_val[c],   errors="ignore")

            dt_cols = X_train.select_dtypes(
                include=["datetime64[ns]", "datetime64[ms]", "datetimetz"]
            ).columns.tolist()

            # 3) Define datetime feature extractors (day, month → cosine)
            def extract_day_month(df: pd.DataFrame) -> pd.DataFrame:
                out = pd.DataFrame(index=df.index)
                for col in df.columns:
                    s = pd.to_datetime(df[col], errors="coerce")
                    out[f"{col}_day"]   = s.dt.day.fillna(0)
                    out[f"{col}_month"] = s.dt.month.fillna(0)
                return out

            def cosine_transform(df: pd.DataFrame) -> pd.DataFrame:
                g = df.copy()
                for c in g.columns:
                    if c.endswith("_day"):
                        g[c] = np.cos(2 * np.pi * (g[c] - 1) / 31.0)
                    elif c.endswith("_month"):
                        g[c] = np.cos(2 * np.pi * (g[c] - 1) / 12.0)
                return g

            extractor = FunctionTransformer(extract_day_month, feature_names_out="one-to-one")
            cosiner   = FunctionTransformer(cosine_transform,   feature_names_out="one-to-one")

            # 4) Build processing pipeline
            numeric_tf = Pipeline([
                ("scale", StandardScaler())
            ])

            categorical_tf = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scale",  StandardScaler(with_mean=False))
            ])

            datetime_tf = Pipeline([
                ("extract", extractor),
                ("cosine",  cosiner),
                ("scale",   StandardScaler())
            ])

            processing = ColumnTransformer(
                transformers=[
                    ("num", numeric_tf, num_cols),
                    ("cat", categorical_tf, cat_cols),
                    ("dt",  datetime_tf, dt_cols),
                ],
                remainder="drop"
            )

            # 5) Build full model pipeline (processing + estimator)
            full_pipe = Pipeline([
                ("preprocess", processing),
                ("est", RandomForestRegressor(random_state=42))
            ])

            # 6) Grid Search on the FULL pipeline (fit ONLY after combining)
            param_grid = {
                "est__n_estimators": [100, 200, 400],
                "est__max_depth": [None, 10, 20],
                "est__min_samples_split": [2, 5, 10]
            }

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            search = GridSearchCV(
                estimator=full_pipe,
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=-1,
                verbose=1
            )

            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_cv_rmse = -search.best_score_
            print(f"Best CV RMSE: {best_cv_rmse:.3f}")
            print(f"Best params: {search.best_params_}")

            # 7) Evaluate best pipeline on validation
            val_pred = best_pipe.predict(X_val)
            val_rmse = mean_squared_error(y_val, val_pred, squared=False)
            print(f"Validation RMSE (best pipeline): {val_rmse:.3f}")

            # 8) Persist the best fitted pipeline (processing + model)
            Path("./artifacts").mkdir(parents=True, exist_ok=True)
            joblib.dump(best_pipe, "./artifacts/best_model_regression.joblib")
            print("Saved: ./artifacts/best_model_regression.joblib")
            """
        ]
    )
    reason: str = Field(
        ...,
        description="Reason for this step.",
        examples=[
            "Different algorithms have varying strengths; selection impacts performance.",
            "Training is essential to learn patterns from data.",
            "Optimal hyperparameters can significantly improve model performance.",
            "Evaluation ensures the model meets the desired performance criteria."
        ]
    )

class Metric(BaseModel):
    name: str = Field(
        ...,
        description="Name of the evaluation metric.",
        examples=["accuracy", "precision", "recall", "F1-score", "RMSE"]
    )
    value: float = Field(
        ...,
        description="Value of the evaluation metric.",
        examples=[0.85, 0.92, 0.78, 0.88, 5.67]
    )

class ModelingOutput(BaseModel):
    best_model: str = Field(
        ...,
        description="Description of the best-performing model."
    )
    metrics: Metric = Field(
        ...,
        description="Performance metrics of the best model."
    )

def execute_data_scientist_step(
    step: DataScientistStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate data scientist tasks to the Coder agent.
    """
    context_variables["current_agent"] = "DataScientist"
    return ReplyResult(
        message=f"Please write Python code to execute this data scientist step:\n{step.step_description} - {step.instruction}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_scientist_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Data science tasks are complete.",
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
    )

class DataScientist(ConversableAgent):
    def __init__(self):
        super().__init__(
            name="DataScientist",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                parallel_tool_calls=False,
                temperature=0.3,
            ),
            human_input_mode="NEVER",
            code_execution_config=False,
            update_agent_state_before_reply=[
                UpdateSystemMessage( 
                    """
                    You are a Data Scientist.
                    Your role is to design, build, and evaluate machine learning models to achieve {objective} for a {problem_type} task. You translate business and 
                    analytical goals into concrete modeling strategies, ensuring results are accurate, explainable, and aligned with stakeholder expectations.

                    Key Responsibilities:
                    - Model selection: Choose appropriate algorithms based on the problem type (e.g., regression, classification, clustering).
                    - Model training: Train models using the end-to-end model pipeline, ensuring proper validation techniques (e.g., cross-validation).
                    - Hyperparameter tuning: Optimize model performance through systematic hyperparameter tuning (e.g., grid search, random search).
                    - Model evaluation: Assess models using relevant metrics (e.g., accuracy, RMSE, F1-score) and validate against business objectives.

                    Workflow:
                    1. Review the business objectives and problem type provided by the BusinessAnalyst.
                    2. For each modeling or evaluation step, call execute_data_scientist_step to delegate implementation to the Coder agent.
                    3. Train, tune, and evaluate models iteratively until performance criteria are met.
                    4. Summarize the best model, including metrics, parameters, and interpretability detail insights of variables related to the best models.
                    5. When modeling and evaluation are complete, call complete_modeling_task with the final model artifacts, metrics, and summary report.

                    Rules:
                    - Do not recommend any thing to Business Translator, only focus on modeling tasks. 
                    - Do not perform prediction for user data point.
                """
                )
            ],
            functions=[execute_data_scientist_step]
        )