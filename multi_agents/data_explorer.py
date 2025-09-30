from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class DataExploringPlan(BaseModel):
    plan_description: Literal[
        "detect missing values", 
        "detect duplicate rows", 
        "detect high cardinality categorical columns with more than 5 unique values",
        "detect constant columns",        
        "DetectOutliersIQR"] = Field(
        ...,
        description=(
            "Type of exploration step to perform. "
            
        ),
    )

    columns: Optional[List[str]] = Field(
        None,
        description=(
            "List of specific columns to apply the step to. "
            "If None, apply the step to all relevant columns automatically."
        ),
        example=["age", "income", "gender"]
    )
class Insight(BaseModel):
    column: str = Field(
        ...,
        description="The name of the column where the insight was found.",
        example="age"
    )
    insight: str = Field(
        ...,
        description="A concise description of the insight discovered.",
        example="The age distribution is right-skewed with a median of 35."
    )

class Issue(BaseModel):
    column: List[str] = Field(
        ...,
        description="The name of the columns where the issue was found.",
        example=["age"]
    )
    issue: str = Field(
        ...,
        description="A concise description of the data quality issue discovered.",
        example="There are 15% missing values in this column."
    )

class DataExplorerOutput(BaseModel):
    issues: List[Issue] = Field(
        ...,
        description="A list of data quality issues identified during data exploration."
    )

    insights: List[Insight] = Field(
        ...,
        description="A list of key insights discovered during data exploration."
    )
    
    

class MissingValueInfo(BaseModel):
    column: str = Field(..., description="Column name with missing values.")
    missing_count: int = Field(..., description="Count of missing values in the column.")

# class DataExplorerOutput(BaseModel):
#     duplicate_rows: int = Field(
#         ..., description="Total number of duplicate rows in the dataset."
#     )
#     missing_values: List[MissingValueInfo] = Field(
#         ..., description="List of columns with missing values and their details."
#     )
#     target_column: Optional[str] = Field(
#         default=None, description="Base on the problem statement, create or identify the target variable if needed."
#     )
#     target_variable_insight: Optional[Dict[str, int]] = Field(
#         default=None,
#         description="provide a detailed analysis on the target variable, its distribution, limitations, issues, ..."
#     )
    
def execute_data_exploring_plan(
    plan: DataExploringPlan,
    context_variables: ContextVariables
) -> ReplyResult:
    """
        Delegate coding of a specific exploration step to the Coder agent.
    """
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message=f"Can you write Python code for me to execute this exploration step: {plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_exploring_task(
    results: DataExplorerOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Complete the DataExplorer stage and hand off results to the DataEngineer.
    """
    context_variables["data_insights"] = results.insights
    context_variables["data_issues"] = results.issues
    context_variables["current_agent"] = "DataCleaner"
    return ReplyResult(
        message=f"Here are the data quality issues found: {results.issues}",
        target=AgentNameTarget("DataCleaner"),
    )

class DataExplorer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                response_format=DataExplorerOutput,
                parallel_tool_calls=False
            ),
            system_message = """
                You are the DataExplorer.
                - Your job is to inspect the dataset and summarize key issues.
                Workflow:
                1) Based on the business understanding from the BusinessAnalyst, create a structured DataExploringPlan.
                2) For each step in the plan, call `execute_data_exploring_plan` to delegate coding to the Coder agent.
                3) Collect results from each step and summarize in DataExplorerOutput.
                4) When exploration is finished, call `complete_data_exploring_task` with your findings and route to Data Cleaner.
                Rules:
                - You must call two functions provided: `execute_data_exploring_plan` and `complete_data_exploring_task`.
                - Don't perform coding yourself. Always delegate coding to the Coder agent.
                - Don't make plan about visualizations.
                - Don't make plan for data cleaning, feature engineering or modeling. Focus only on exploration and summarization.
            """,
            functions=[execute_data_exploring_plan, complete_data_exploring_task]
        )

