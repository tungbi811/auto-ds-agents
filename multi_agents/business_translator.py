from autogen import AssistantAgent
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ProblemsSection(BaseModel):
    """
    Problems - Description: Detailed description of the business problems
    """
    description: str = Field(
        ..., description="Detailed description of the business problems"
    )
    objectives: Optional[List[str]] = Field(
        None, description="Optional list of clarified business objectives or key questions"
    )


class DataExplorationInsightSection(BaseModel):
    """
    Data Exploration Insight - Description: Detailed description of EDA insights that support solving
    the business problems (e.g., business trends, key drivers, potential impact factors)
    """
    description: str = Field(
        ..., description="Narrative summary of EDA insights supporting the business problems"
    )
    trends: Optional[List[str]] = Field(
        None, description="Notable business/data trends observed in EDA"
    )
    key_drivers: Optional[List[str]] = Field(
        None, description="Variables/features most associated with the business outcome"
    )
    impact_factors: Optional[List[str]] = Field(
        None, description="Potential impact factors (market, seasonality, segments, constraints)"
    )
    evidence: Optional[List[str]] = Field(
        None, description="Pointers to evidence (tables/plots/metrics) that support these insights"
    )


class BusinessSuggestionItem(BaseModel):
    """
    Business Suggestions and Rationale - one actionable recommendation
    """
    suggestion: str = Field(
        ..., description="Business-oriented recommendation to address the problems"
    )
    rationale: str = Field(
        ..., description="Reasoning grounded in Data Analyst and Data Scientist findings, translated to business value"
    )
    expected_value: Optional[str] = Field(
        None, description="Expected business impact (e.g., revenue uplift, cost reduction, risk mitigation)"
    )
    stakeholders_impacted: Optional[List[str]] = Field(
        None, description="Stakeholders affected by this suggestion"
    )
    priority: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Execution priority"
    )
    prerequisites: Optional[List[str]] = Field(
        None, description="Dependencies, data/process prerequisites to implement"
    )
    risks: Optional[List[str]] = Field(
        None, description="Potential risks or trade-offs when applying the suggestion"
    )
    kpis_to_monitor: Optional[List[str]] = Field(
        None, description="KPIs to track to validate impact in production"
    )
    timeframe: Optional[str] = Field(
        None, description="Indicative timeline or phases to implement (e.g., short/medium/long term)"
    )


class BusinessTranslatorReport(BaseModel):
    """
    Main structured output for the Business Translator Agent
    """
    problems: ProblemsSection = Field(
        ..., description="Detailed description of the business problems"
    )
    data_exploration_insight: DataExplorationInsightSection = Field(
        ..., description="EDA insights that support solving the business problems"
    )
    business_suggestions_and_rationale: List[BusinessSuggestionItem] = Field(
        ..., description="Business-oriented recommendations with rationale grounded in analyst & DS reports"
    )



class BusinessTranslator(AssistantAgent):
    def __init__(self, llm_config):
        super().__init__(
            name="BusinessTranslator",
            llm_config=llm_config,
            system_message="""
                You are the Business Translation Agent.
                Your job is to explain technical results in clear business terms.

                Tasks:
                - Summarize the model results in plain language.
                - Map results back to business objectives and KPIs.
                - Highlight strengths, limitations, risks, and trade-offs.
                - Provide recommendations for deployment or iteration.
                - Suggest monitoring and governance requirements.

                Rules:
                - Write in simple language for non-technical stakeholders.
                - End your FINAL message with <END_OF_WORKFLOW>.
            """
        )
