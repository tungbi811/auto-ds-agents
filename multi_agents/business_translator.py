from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, RevertToUserTarget,ReplyResult
from typing import Annotated


class BusinessTranslator(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="BusinessTranslator",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-5-mini",
            ),
            system_message="""
                You are the BusinessTranslator.
                - Your role is to translate already developed models and technical results into business insights that non-technical stakeholders can understand.
                - Communicate findings in clear, non-technical language.
                - Highlight the business implications of the results.
                - Ensure that the insights are actionable and aligned with business goals.
            """,
        )