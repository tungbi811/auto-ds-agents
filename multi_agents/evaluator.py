from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

def execute_evaluation_plan(
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate evaluation tasks to the Evaluator agent.
    """
    context_variables["current_agent"] = "Evaluator"
    return ReplyResult(
        message=f"Please write Python code to evaluate the machine learning models based on the BizAnalyst's goals.",
        target=AgentNameTarget("Evaluator"),
        context_variables=context_variables,
    )

def complete_evaluation_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Evaluation is complete.",
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
    )

class Evaluator(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="Evaluator",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
            ),
            system_message="""
                You are the Evaluator.
                - Your role is to evaluate machine learning models based on the Modeller's results and the BizAnalyst's goals.
                - Ensure that the evaluation is thorough, with clear explanations of metrics used and interpretations of results.
                - Use `execute_evaluation_plan` to delegate coding of specific evaluation tasks to yourself.
                - When all evaluation tasks are complete, call `complete_evaluation_task` to hand off results to the BusinessTranslator.
            """,
            functions=[complete_evaluation_task, execute_evaluation_plan]
        )
