from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

def execute_modeling_plan(
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate modeling tasks to the Coder agent.
    """
    return ReplyResult(
        message=f"Please write Python code to build, evaluate, and select machine learning models based on the processed data and the BizAnalyst's goals.",
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

class Modeler(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="Modeler",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                parallel_tool_calls=False,
                temperature=0.3,
            ),
            system_message="""
                You are the Modeler.
                - Your role is to build, evaluate, and select machine learning models based on the FeatureEngineer's processed data and the BizAnalyst's goals.
                - Ensure that the models are well-documented, with clear explanations of choices made during the modeling process.
                - Use `execute_modeling_plan` to delegate coding of specific modeling tasks to the Coder agent.
                - When all modeling tasks are complete, call `complete_modeling_task` to hand off results to the Evaluator.
            """,
            functions=[complete_modeling_task, execute_modeling_plan]
        )