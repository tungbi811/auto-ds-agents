from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, RevertToUserTarget,ReplyResult
from pydantic import BaseModel, Field

class BusinessTranslationStep(BaseModel):
    instruction: str = Field(
        ...,
        description="A specific instruction or task that needs to be solved to answer stakeholder expectations.",
        example=[
            "Calculate the average customer lifetime value for each customer segment.",
            "Identify the top 5 factors contributing to customer churn.",
        ]
    )

def execute_business_translation_step(
    step: BusinessTranslationStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Translate a high-level business task into specific data science objectives.
    Example task: 'Increase customer retention by 10% over the next quarter.'
    """
    context_variables["current_agent"] = "BusinessTranslator"
    return ReplyResult(
        message=f"Write python code to achieve the following task:\n{step}",
        target=AgentNameTarget("BusinessAnalyst"),
        context_variables=context_variables,
    )

def complete_business_translation_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Business translation is complete.",
        target=RevertToUserTarget(),
        context_variables=context_variables,
    )

class BusinessTranslator(ConversableAgent):
    def __init__(self):
        super().__init__(
            name="BusinessTranslator",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-5-mini",
                # parallel_tool_calls=False,
                # temperature=0.5,
            ),
            update_agent_state_before_reply=UpdateSystemMessage(
                """
                    You are the BusinessTranslator.
                    Your role is to interpret analytical results and translate them into actionable business insights and strategic plans tailored to stakeholder needs.
                    You ensure that technical findings are clearly connected to business objectives, stakeholder expectations, and measurable outcomes.

                    Stakeholder Expectations:
                    {stakeholders_expectations}

                    Key Responsibilities:
                    - Understand the business context and intent behind the research question.
                    - Interpret technical or statistical findings in plain business language, emphasizing their practical meaning and implications.
                    - Translate insights into concrete, stakeholder-specific action plans that align with strategic goals and operational needs.
                    - Ensure recommendations are feasible, data-driven, and directly tied to the stakeholder_expectations provided.
                    - Quantify expected outcomes where possible (e.g., potential revenue growth, efficiency improvement, customer engagement uplift).

                    Workflow:
                    1. Analyze stakeholder_expectations to understand desired outcomes and key performance indicators.
                    2. Translate these findings into actionable business implications for each stakeholder group.

                    Rules:
                    - Do not include technical details such as algorithms, data preprocessing, or modeling methods.
                    - Focus on clarity, relevance, and real-world business applicability.
                    - Ensure that every recommendation or action plan ties back to both the research question and stakeholder_expectations.
                    - Use clear, persuasive, and business-oriented language suitable for decision-makers.
                """
            ),
            functions = [complete_business_translation_task]
        )