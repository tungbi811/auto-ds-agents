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
            "(1) load the cleaned splits saved by the Data Engineer, "
            "(2) build a processing pipeline with categorical → OneHotEncoder → StandardScaler; numerical → StandardScaler; "
            "datetime → extract [day, month] → cosine transform → StandardScaler, "
            "(3) for each candidate model, create ONE sklearn Pipeline [('preprocess', processing_pipeline), ('est', estimator)], "
            "(4) NEVER call fit() on the preprocessing pipeline by itself; call fit() ONLY on the combined pipeline using the train_cleaned split"
            "(5) select the appropriate cross-validation strategy (e.g., StratifiedKFold/KFold/TimeSeriesSplit) and scoring metric based on the {problem_type}, "
            "(6) ALWAYS fit on the train_cleaned split you loaded (no validation/test leakage), then evaluate on val_cleaned and optionally test_cleaned, "
            "and (7) persist the best fitted pipeline (processing + estimator) to the specified artifact path."
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