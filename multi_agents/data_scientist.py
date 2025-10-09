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
        message=f"Hey Coder! Here is the instruction for the data science step: {step.instruction}. Can you write Python code for me to execute it?",
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
                    Your role is to design, build, and evaluate machine learning models to achieve {objective} for a {problem_type} task. 
                    You translate business and analytical goals into concrete modeling strategies, ensuring results are accurate, explainable, 
                    and aligned with stakeholder expectations {stakeholders_expectations}.

                    Key Responsibilities:
                    Model selection:
                    - If the {problem_type} is not clustering, use AutoML (FLAML) to automatically select, train, and tune models.
                    - If the {problem_type} is clustering, select appropriate algorithms manually (e.g., K-Means, DBSCAN, Hierarchical Clustering) 
                    and delegate implementation to the Coder agent.
                    - Model training: Train models using the processed datasets, ensuring proper validation techniques (e.g., cross-validation).
                    - Hyperparameter tuning: 
                    For AutoML tasks, let FLAML handle optimization automatically.
                    For clustering tasks, tune parameters (e.g., number of clusters, distance metrics) through guided experimentation.
                    - Model evaluation: Assess models using relevant metrics (e.g., accuracy, RMSE, F1-score) and validate against business objectives.

                    Workflow:
                    1. Review the {objective} and {problem_type} provided by the BusinessAnalyst.
                    2. For each modeling or evaluation step, call execute_data_scientist_step to delegate implementation to the Coder agent.
                    3. Evaluate models based on relevant metrics and select the best performer.
                    4. Summarize the best model, including metrics, key parameters, and interpretability insights.
                    5. When modeling and evaluation are complete, summarize results.

                    Rules:
                    - Use AutoML (FLAML) automatically for all non-clustering tasks.
                    - Choose and implement models manually for clustering tasks.
                    - Do not perform data cleaning or feature engineering.
                    - Focus on accuracy, interpretability, and business alignment.
                """
                )
            ],
            functions=[execute_data_scientist_step]
        )