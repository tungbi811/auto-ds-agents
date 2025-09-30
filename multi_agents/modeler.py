from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field
from typing import Optional

# ---------- Structured Output ----------
class BestModelInfo(BaseModel):
    model_name: str = Field(..., description="Name of the best selected model.")
    metrics: dict = Field(..., description="Performance metrics of the best model.")
    pipeline_path: str = Field(..., description="Relative path where the best model pipeline is saved, e.g. 'model/best_model.pkl'.")

# ---------- Functions ----------
def execute_modeling_plan(
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate modeling tasks to the Coder agent.
    """
    context_variables["current_agent"] = "Modeler"
    return ReplyResult(
        message=(
            "Please write Python code to:\n"
            "1. Build, evaluate, and select machine learning models based on the processed data and BizAnalyst's goals.\n"
            "2. Identify the best-performing model.\n"
            "3. Save its pipeline using joblib or pickle to the folder 'model/' "
            "with filename 'best_model.pkl'.\n"
            "4. Return structured output according to BestModelInfo schema."
        ),
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_modeling_task(
    context_variables: ContextVariables,
    best_model_info: Optional[BestModelInfo] = None,
) -> ReplyResult:
    """
    Pass modeling completion message and best model info to Business Translator.
    """
    context_variables["current_agent"] = "BusinessTranslator"
    return ReplyResult(
        message="Modeling is complete.",
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
        structured_output=best_model_info.dict() if best_model_info else None,
    )

# ---------- Modeler Agent ----------
class Modeler(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="Modeler",
            llm_config=LLMConfig(
                api_type="openai",
                model="gpt-4.1-mini",
                parallel_tool_calls=False,
                temperature=0.3,
            ),
            system_message="""
                You are the Modeler.
                - Your role is to build, evaluate, and select machine learning models based on the FeatureEngineer's processed data and the BizAnalyst's goals.
                - Ensure that the models are well-documented, with clear explanations of choices made during the modeling process.
                - Use `execute_modeling_plan` to delegate coding of specific modeling tasks to the Coder agent.
                - When the best model is selected and saved to 'model/best_model.pkl', call `complete_modeling_task`
                  and provide structured output in BestModelInfo format.
                
                Rules:
                - Always provide BestModelInfo structured output at the end.
                - You must use 2 provided functions: `execute_modeling_plan`, `complete_modeling_task`.
            """,
            functions=[complete_modeling_task, execute_modeling_plan]
        )
