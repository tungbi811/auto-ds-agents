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

def answer_business_translator(
    response: str,
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=response,
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
    )

class DataScientist(ConversableAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type="openai",
            model="gpt-4.1-mini",
            temperature=0.3,
            stream=False,
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
                    Your role is to answer Business Translator task instructions by breaking them down into specific, manageable data science steps.
                    Use execute_data_scientist_step function to delegate coding tasks to the Coder agent.
                    Use answer_business_translator function to give feedback to the Business Translator.
                """
                )
            ],
            functions=[execute_data_scientist_step, answer_business_translator]
        )