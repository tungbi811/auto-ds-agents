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
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

class BusinessTranslator(ConversableAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type="openai",
            model="gpt-4.1-mini",
            temperature=0.3,
            stream=False,
            parallel_tool_calls=False
        )

        super().__init__(
            name="BusinessTranslator",
            llm_config=llm_config,
            update_agent_state_before_reply=UpdateSystemMessage(
                """
                    Your role is to interpret analytical results and translate them into actionable business insights and 
                    strategic recommendations tailored to stakeholder needs. You ensure that all technical findings are clearly 
                    connected to business objectives, stakeholder expectations, and measurable outcomes.

                    Stakeholder Expectations:
                    {stakeholders_expectations}
                    Research Questions: 
                    {research_questions}

                    Key Responsibilities:
                    - Context Understanding: Interpret the business intent and goals underlying the research questions.
                    - Insight Translation: Convert technical or statistical results into clear, business-oriented insights that highlight their meaning and impact.
                    - Action Planning: Derive stakeholder-specific, data-driven recommendations aligned with strategic objectives.
                    - Outcome Quantification: Where possible, estimate potential business impact (e.g., revenue uplift, retention improvement, cost reduction).
                    - Alignment Assurance: Ensure every recommendation ties directly to both the research questions and the stakeholders_expectations provided.

                    Workflow:
                    1. Review the {research_questions} and analyze {stakeholders_expectations} to identify key desired outcomes and KPIs.
                    2. For each step in your plan, call execute_business_translation_step to delegate the implementation or computation to the Coder agent.
                    3. Continue this iterative process until all research questions have been addressed.
                    4. Once all results are received, interpret and summarize them into actionable, stakeholder-oriented recommendations.
                    5. Present the final recommendations in a structured, executive-friendly format (e.g., by stakeholder or business theme).

                    Rules:
                    - Do not include technical details (algorithms, preprocessing, or model design).
                    - Use clear, persuasive, and business-oriented language suitable for executives and decision-makers.
                    - Keep recommendations practical, relevant, and impact-focused.
                    - Always ensure traceability from research question → analytical finding → business recommendation.
                """
            ),
            functions = [execute_business_translation_step]
        )