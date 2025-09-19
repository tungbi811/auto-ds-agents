from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from utils.utils import request_clarification

class Stakeholder(BaseModel):
    name: str = Field(..., description="Stakeholder name or group")
    role: Optional[str] = Field(None, description="Role in the project")
    interests: Optional[str] = Field(None, description="Benefits/concerns, what they care about")
    impact_level: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Expected impact level on this stakeholder"
    )

class TargetVariableProposal(BaseModel):
    name: str = Field(..., description="Candidate target variable")
    rationale: Optional[str] = Field(None, description="Why this target fits the business goal")
    dtype: Optional[Literal["binary", "categorical", "continuous", "ordinal"]] = Field(
        None, description="Data type of the candidate target"
    )
    notes: Optional[str] = Field(None, description="Any data availability or definition caveats")

class EvaluationMetric(BaseModel):
    name: str = Field(..., description="Metric name (e.g., RMSE, MAE, F1, ROC-AUC, Recall)")
    definition: Optional[str] = Field(None, description="What the metric measures")
    preferred_direction: Optional[Literal["maximize", "minimize"]] = Field(
        None, description="Optimization direction for this metric"
    )
    rationale: Optional[str] = Field(None, description="Why this metric is appropriate for the problem")

class OptimizeMetric(BaseModel):
    metric_name: str = Field(..., description="The single most important metric to optimize")
    why: str = Field(..., description="Reason for prioritizing this metric")
    direction: Literal["maximize", "minimize"] = Field(..., description="Optimization direction")

class ImpactItem(BaseModel):
    stakeholder: str = Field(..., description="Stakeholder name or group")
    impact_description: str = Field(..., description="Type/level of impact for this stakeholder")
    impact_level: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Severity/importance of the impact"
    )

class RiskItem(BaseModel):
    kind: Literal["data", "model", "business", "regulatory", "operational", "other"] = Field(
        ..., description="Risk/assumption/limitation category"
    )
    description: str = Field(..., description="What is the risk/assumption/limitation?")
    severity: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Severity or importance"
    )
    mitigation: Optional[str] = Field(None, description="Proposed mitigation or monitoring approach")

class Recommendation(BaseModel):
    action: str = Field(..., description="Actionable suggestion")
    rationale: Optional[str] = Field(None, description="Business/technical reasoning for this action")
    priority: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Execution priority"
    )
    dependencies: Optional[List[str]] = Field(
        None, description="Prerequisites or blocking items"
    )


# --- Main BA structured output ---

class BusinessAnalysisReport(BaseModel):
    business_problems: str = Field(..., description="Detailed description of business problems")

    stakeholders: List[Stakeholder] = Field(
        default_factory=list,
        description="List and description of each stakeholder and their role/benefit"
    )

    field_of_problems: str = Field(..., description="Description of the business domain")

    problem_type: Literal[
        "regression", "classification", "clustering", "forecasting",
        "recommendation", "ranking", "nlp", "cv", "other"
    ] = Field(..., description="Problem type")

    goal_of_project: str = Field(..., description="Quantitative/qualitative project goal")

    potential_target_variable: List[TargetVariableProposal] = Field(
        default_factory=list,
        description="Candidate target variable(s) and rationale"
    )

    evaluation_metrics: List[EvaluationMetric] = Field(
        default_factory=list,
        description="List and explanation of suitable metrics"
    )

    most_important_evaluation_value: OptimizeMetric = Field(
        ..., description="Which metric to optimize and why"
    )

    impact_of_results_to_stakeholders: List[ImpactItem] = Field(
        default_factory=list,
        description="Explain the level and type of impact to stakeholders"
    )

    risks_assumptions_limitations: List[RiskItem] = Field(
        default_factory=list,
        description="Key risks, assumptions, and limitations"
    )

    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Initial actionable suggestions"
    )
    # )

class BusinessAnalyst(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="BusinessAnalyst",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                response_format=BusinessAnalysisReport
            ),
            system_message="""
                You are the Business Analyst Agent for the CRISP-DM Business Understanding phase.
                Your task is to take the user’s request and return a JSON object that strictly follows the BizAnalystOutput schema.
                - Always return only valid JSON that matches the schema exactly.
                - If the user does not provide enough information, use the request_clarification tool to ask clear, targeted a follow-up question until you can complete the required output (only one question at a time). Once sufficient details are gathered, return the structured JSON.
                - Keep answers clear, concise, and focused on business value.
                - Do not propose algorithms or implementation details.
            """,
            functions = [request_clarification]
        )