from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field

class ModelingStep(BaseModel):
    step_description: str = Field(
        ...,
        description="Modeling step to perform.",
        examples=[
            "Select algorithms",
            "Train models",
            "Hyperparameter tuning",
            "Evaluate models"
        ]
    )
    action: str = Field(
        ...,
        description="Action to take for this step.",
        examples=[
            "Choose algorithms suitable for the problem type (e.g., regression, classification, clustering).",
            "Train selected algorithms on the training dataset.",
            "Use grid search or random search to find optimal hyperparameters.",
            "Evaluate models using appropriate metrics (e.g., accuracy, RMSE, F1-score) on validation dataset."
        ]
    )
    suggestion: str = Field(
        ...,
        description="How to perform this step.",
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

def execute_modeling_plan(
    step: ModelingStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate modeling tasks to the Coder agent.
    """
    return ReplyResult(
        message=f"Please write Python code to execute this data cleaning step:\n{step.step_description} - {step.action} - {step.suggestion}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_modeling_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Modeling is complete.",
        target=AgentNameTarget("Evaluator"),
        context_variables=context_variables,
    )

class Modeler(ConversableAgent):
    def __init__(self):
        super().__init__(
            name="Modeler",
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
                    You are the Modeler.
                    Your role is to design, build, and evaluate machine learning models to achieve {objective} for a {problem_type} task. You translate business and 
                    analytical goals into concrete modeling strategies, ensuring results are accurate, explainable, and aligned with stakeholder expectations.

                    Key Responsibilities:
                    - Model selection: Choose appropriate algorithms based on the problem type (e.g., regression, classification, clustering).
                    - Model training: Train models using the processed datasets, ensuring proper validation techniques (e.g., cross-validation).
                    - Hyperparameter tuning: Optimize model performance through systematic hyperparameter tuning (e.g., grid search, random search).
                    - Model evaluation: Assess models using relevant metrics (e.g., accuracy, RMSE, F1-score) and validate against business objectives.

                    Workflow:
                    1. Review the business objectives and problem type provided by the BusinessAnalyst.
                    2. For each modeling or evaluation step, call execute_modeling_step to delegate implementation to the Coder agent.
                    3. Train, tune, and evaluate models iteratively until performance criteria are met.
                    4. Summarize the best model, including metrics, parameters, and interpretability insights.
                    5. When modeling and evaluation are complete, call complete_modeling_task with the final model artifacts, metrics, and summary report.
                """
                )
            ],
            functions=[complete_modeling_task, execute_modeling_plan]
        )