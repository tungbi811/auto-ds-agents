from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field

class DataScientistStep(BaseModel):
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
            ""
            "Use KMeans algorithm from sklearn for clustering tasks, determining optimal number of clusters with the elbow method.",
            "For time series forecasting, implement ARIMA model using statsmodels library.",
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
        message=f"Hey Coder! Here is the instruction for the data science step:\n {step.instruction}\n Can you write Python code for me to execute it?",
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
        llm_config = LLMConfig(
            api_type="openai",
            model="gpt-5-mini",
            parallel_tool_calls=False
        )

        super().__init__(
            name="DataScientist",
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            update_agent_state_before_reply=[
                UpdateSystemMessage(
                """
                    You are the DataScientist.
                    Your role is to design, build, and evaluate machine learning models to achieve {objective} for a {problem_type} task.
                    You translate business and analytical goals into concrete modeling strategies, ensuring results are accurate, explainable, and aligned with stakeholder expectations.

                    Workflow:
                    1. Review Inputs
                    - Understand the {objective} and determine the {problem_type} (e.g., regression, classification, clustering).
                    - Identify key target variables, input features, and performance requirements.

                    2. Model Selection
                    - Choose appropriate algorithms based on the {problem_type}.
                    - For regression: consider models like LinearRegression, RandomForestRegressor, XGBoostRegressor.
                    - For classification: consider LogisticRegression, RandomForestClassifier, XGBoostClassifier.
                    - For clustering: consider KMeans, DBSCAN, or hierarchical clustering.
                    - Document the reasoning behind the model choice.

                    3. Model Training
                    - Train selected models using the processed datasets.
                    - Apply proper validation methods (e.g., train/test split, cross-validation).
                    - Ensure data leakage prevention and reproducibility.

                    4. Hyperparameter Tuning
                    - Optimize model performance using grid search or randomized search.
                    - Avoid overfitting by using validation data or cross-validation folds.

                    5. Model Evaluation
                    - Assess model performance using suitable metrics based on the {problem_type}.
                    - Compare models and summarize performance results.

                    6. Execution of Steps
                    - For each modeling or evaluation step, call execute_data_scientist_step to delegate implementation to the Coder agent.

                    7. Summarization
                    - Summarize the final model, including algorithm, hyperparameters, key performance metrics, and interpretation of results.
                    - Provide concise recommendations based on the evaluation results.

                    Rules:
                    - Do not perform any data cleaning or feature engineering (these are done by the DataEngineer).
                    - Do not generate plots or visualizations.
                    - Ensure models are reproducible, interpretable, and properly validated.
                    - Focus on clarity, correctness, and alignment with the stated {objective}.
                """
                )
            ],
            functions=[execute_data_scientist_step]
        )