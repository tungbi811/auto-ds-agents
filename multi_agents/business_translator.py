from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, RevertToUserTarget,ReplyResult
from pydantic import BaseModel, Field


class BusinessTranslatorStep(BaseModel):
    step_description: str = Field(
        ...,
        description=(
            "Type of business translation step to perform."
        ),
        examples=[
            "Interpret analytical results and connect them to business objectives."
        ]
    )

    instruction: str = Field(
        ...,
        description="Detailed instructions on what and how to perform this step.",
        examples=[
            (
                "1. Review the user’s question to determine if it requires a concise or detailed analytical explanation.\n"
                "2. Examine outputs from all previous agents to fully understand the analytical context.\n"
                "3. Summarize analytical and model findings, identifying key variables that influence the target outcome.\n"
                "4. Translate these findings into clear business insights.\n"
                "5. If detailed interpretation is needed, provide business action recommendations — explain their rationale, "
                "associated risks, and alignment with business goals."
            )
        ]
    )

    reason: str = Field(
        ...,
        description="Purpose and rationale for performing this step.",
        examples=[
            (
                "Ensure that analytical insights are directly tied to the business question, clearly explained, and "
                "supported by evidence. This step helps the user understand the reasoning behind conclusions and "
                "recognize the potential risks and strategic implications of recommended business actions."
            )
        ]
    )

    examples: str = Field(
        ...,
        description="Illustrative examples of how this step applies across different analytical tasks.",
        examples=[
            (
                "• Clustering: Explain how clusters are formed and its deatail information based on defining characteristics and analyze the opportunities "
                "and risks of business actions targeting each cluster.\n"
                "• Regression / Time-Series: Identify which variables most influence the target, interpret their business meaning, "
                "and propose actions that could optimize future outcomes.\n"
                "• Classification: Highlight variables that affect the target decision (e.g., Yes/No), clarify positive and negative "
                "drivers, and suggest actions to improve the likelihood of desired results."
            )
        ]
    )



def execute_business_translation_step(
    step: BusinessTranslatorStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Translate a high-level business task into specific data science objectives.
    Example task: 'Increase customer retention by 10% over the next quarter.'
    """
    context_variables["current_agent"] = "BusinessTranslator"
    return ReplyResult(
        message=f"Please translate this business task into specific data science objectives:\n{task}",
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
                # temperature=0.3,
            ),
            update_agent_state_before_reply=UpdateSystemMessage(
            """
            You are the BusinessTranslator.
            Your role is to interpret analytical results and translate them into clear, actionable business insights and strategic recommendations tailored to stakeholder needs.
            You ensure that analytical findings are directly connected to business objectives, stakeholder expectations, and measurable outcomes.

            Stakeholder Expectations:
            {stakeholders_expectations}

            Key Responsibilities:
            - Understand the business context and intent behind the research question.
            - Review analytical outputs provided by other agents to extract key insights that directly address the stakeholder’s question.
            - Communicate findings in plain business language, focusing on their practical meaning and implications.
            - **Clarity and focus** — respond directly to the stakeholder’s intent; avoid technical language or lengthy explanations.
            - **Insight-driven** — highlight only the most essential patterns or segment characteristics that explain the outcome.
            - **Action-oriented** — for each key insight, propose what the business should *do* (e.g., target, invest, optimize, mitigate) and outline potential benefits or risks.
            - **Generalizable** — ensure the output format applies to any business question or dataset (e.g., segmentation, forecasting, classification).
            - **Plain business language** — never mention algorithms, preprocessing steps, or statistical terminology.

            Workflow:
            1. Review stakeholder_expectations to clearly understand the desired business outcomes and key performance indicators.
            2. Examine the analytical results provided by other agents to identify the main findings relevant to the business question.
            3. If insights are incomplete or unclear, request additional analysis or data exploration from technical agents (e.g., Coder or Data Explorer).
            - This may include re-scaling or reconstructing datasets to their original form, re-labelling segments or classes, or applying supplementary analytical methods.
            - The goal is to ensure that the findings provide a complete and interpretable view of each segment, pattern, or business condition.
            4. Once sufficient analytical clarity is achieved, interpret the refined results in business terms.
            5. Translate the findings into actionable insights, clearly outlining implications, opportunities, risks, and recommended actions for decision-makers.
            6. Summarize the final output in a concise business format (e.g., **Key Insights** and **Business Recommendations**) that directly answers the stakeholder’s question.

            Example style:
            - “The analysis identifies three property segments: luxury, affordable, and investment-ready homes.”
            - “Luxury homes show strong price stability but slower turnover; affordable homes attract first-time buyers.”
            - “Recommendation: Focus marketing efforts on affordable homes to drive higher sales volume, while positioning luxury homes through exclusive channels to maximize margins.”

            Rules:
            - Do not include technical details such as algorithms, data preprocessing, or modeling methods.
            - Do not use any technical terms such as “regression,” “cluster,” “p-value,” “confidence interval,” or “feature importance.”
            - A strong understanding of the underlying characteristics provides the foundation for meaningful business insights.
            - Focus on clarity, relevance, and real-world applicability.
            - Ensure that each recommendation or action plan directly aligns with both the research question and stakeholder expectations.
            - Use persuasive, professional, and business-oriented language suitable for decision-makers.
            - Maintain a concise and results-driven tone.
            - Ensure every insight and recommendation connects logically to the question and stakeholder goals.
            """

            ),
        )


