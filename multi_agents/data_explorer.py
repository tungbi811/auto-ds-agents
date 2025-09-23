from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated

class TargetVariableInfo(BaseModel):
    name: str = Field(..., description="Name of the target variable")
    dtype: str = Field(..., description="Data type of target variable (Binary, Categorical, Continuous, etc.)")


class DistinctVariableInfo(BaseModel):
    name: str = Field(..., description="Variable that contains only distinct values (e.g., ID, Customer_ID)")
    notes: Optional[str] = Field(None, description="Additional notes about why this variable is distinct")


class UnidentifiedValueIssue(BaseModel):
    column: str = Field(..., description="Variable that contains unidentified values")
    details: Optional[str] = Field(None, description="Details about the unidentified values issue")
    proposed_solution: Optional[str] = Field(None, description="Proposed solution to handle unidentified values")


class OutlierInfo(BaseModel):
    column: str = Field(..., description="Variable name where outliers occur")
    outlier_type: str = Field(..., description="Type of outliers: bad (harmful) or good (meaningful)")
    count: int = Field(..., description="Number of outliers detected")
    insights: Optional[str] = Field(None, description="Insights about the outliers and their impact")
    proposed_solution: Optional[str] = Field(None, description="Suggested solution for outliers")


class DuplicateInfo(BaseModel):
    count: int = Field(..., description="Number of duplicated rows/records")
    insights: Optional[str] = Field(None, description="Explanation of problems caused by duplicates")
    proposed_solution: Optional[str] = Field(None, description="Suggested approaches to handle duplicates")


class MissingValueInfo(BaseModel):
    column: str = Field(..., description="Variable that contains missing values")
    count: int = Field(..., description="Number of missing values")
    insights: Optional[str] = Field(None, description="Explanation of missing value issues and their potential impact")
    proposed_solution: Optional[str] = Field(None, description="Suggested solution for missing values")


class DistributionInsight(BaseModel):
    column: str = Field(..., description="Variable analyzed")
    description: str = Field(..., description="Summary of distribution and key insights")


class CorrelationInsight(BaseModel):
    variable_x: str = Field(..., description="First variable in the correlation relationship")
    variable_y: str = Field(..., description="Second variable in the correlation relationship")
    strength: Optional[str] = Field(None, description="Strength of correlation (weak, moderate, strong)")
    insights: Optional[str] = Field(None, description="Key insights about the correlation")
    problem: Optional[str] = Field(None, description="Problem identified, if any (e.g., multicollinearity, redundancy)")


class DataExplorerReport(BaseModel):
    target: TargetVariableInfo
    distinct_variables: List[DistinctVariableInfo] = []
    unidentified_values: List[UnidentifiedValueIssue] = []
    outliers: List[OutlierInfo] = []
    duplicates: Optional[DuplicateInfo] = None
    missing_values: List[MissingValueInfo] = []
    distributions: List[DistributionInsight] = []
    correlations: List[CorrelationInsight] = []

    overall_insights: Optional[str] = Field(None, description="Main findings from the EDA relevant to the business problem")
    proposed_next_steps: Optional[List[str]] = Field(None, description="Recommended actions for downstream agents")

def execute_code(
    plan: Annotated[str, "What do you want coder write code for you just one small step don't plan too much"],
    context_variables: ContextVariables) -> ReplyResult:
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message=f"Can you write code for me to execute this plan {plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables
    )

def complete_task(results: DataExplorerReport, context_variables: ContextVariables) -> ReplyResult:
    return ReplyResult(
        message=f"Issue: {results['missing_values']}",
        target=AgentNameTarget("DataEngneer")
    )

class DataExplorer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                response_format=DataExplorerReport,
                parallel_tool_calls=False
            ),
            system_message = """
                You are a Senior Data Analyst with deep expertise in real estate data and data science projects. 
                Your responsibility is perform a thorough exploratory data analysis (EDA) using run code tool that 
                ensures the dataset is well-understood, potential issues are identified, and meaningful insights are 
                generated for the downstream data engineering and modeling phases. Do not code by yourself, coder will write and 
                execute code for you instead, tell coder what you want 
            """,
            functions=[execute_code]
        )

