from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, RevertToUserTarget,ReplyResult
from typing import Annotated

def execute_business_translation_task(
    task: Annotated[str, "The business task or goal to be translated."],
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Translate a high-level business task into specific data science objectives.
    Example task: 'Increase customer retention by 10% over the next quarter.'
    """
    context_variables["current_agent"] = "BusinessTranslator"
    return ReplyResult(
        message=f"Please translate this business task into specific data science objectives:\n{task}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_business_translation_task(
    answer: Annotated[str, "The final answer or summary of user requirements."],
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=answer,
        target=RevertToUserTarget(),
        context_variables=context_variables,
    )

class BusinessTranslator(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="BusinessTranslator",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-5-mini",
                parallel_tool_calls=False,
            ),
            system_message="""
                You are the BusinessTranslator.
                - Your role is to translate already developed models and technical results into business insights that non-technical stakeholders can understand.
                - Communicate findings in clear, non-technical language.
                - Highlight the business implications of the results.
                - Ensure that the insights are actionable and aligned with business goals.
                - Use `execute_business_translation_task` to delegate the translation task to the Coder agent.
                - When the translation is complete, call `complete_business_translation_task` to answer user questions.
            """,
            functions=[execute_business_translation_task, complete_business_translation_task]
        )