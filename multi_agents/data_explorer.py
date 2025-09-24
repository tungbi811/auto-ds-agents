from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal

class IssueItem(BaseModel):
    column: str = Field(..., description="Name of the column with an issue.")
    issue: str = Field(..., description="Description of the issue with this column (e.g., missing values, outliers, all unique, low/high correlation with other column).")

class DataExplorerOutput(BaseModel):
    insights: List[str] = Field(
        ..., description="Key dataset-level insights as concise bullets."
    )
    issues: List[IssueItem] = Field(
        ..., description="Column-level issues found during data exploration."
    )

def execute_data_exploring_plan(
    plan: Annotated[str, "What do you want coder write code for you just one small step don't plan too much"],
    context_variables: ContextVariables) -> ReplyResult:
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message=f"Can you write code for me to execute this plan {plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables
    )

def complete_data_explore_task(results: DataExplorerOutput, context_variables: ContextVariables) -> ReplyResult:
    return ReplyResult(
        message=f"Issue: {results.issues}",
        target=AgentNameTarget("DataEngineer")
    )

class DataExplorer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                response_format=DataExplorerOutput,
                parallel_tool_calls=False
            ),
            system_message = """
                    You are a Senior Data Analyst with deep expertise in real estate data and data science projects.  
                    Your responsibility is to perform a structured exploratory data analysis (EDA) focused on identifying DATA PROBLEMS across ALL variables in the dataset, not just the target.  
                    Follow this process strictly:
                    1. Dataset Understanding
                    - Summarize dataset: number of rows, number of columns, variable types (numeric, categorical, datetime, text).
                    - Identify columns that are distinct identifiers (e.g., ID, Customer_ID).
                    2. Missing Values
                    - For each column: report missing_count and missing_rate.
                    - Only list columns that actually contain missing values.
                    3. Unidentified Values
                    - Detect variables with ambiguous or mixed types.
                    - Provide variable name and sample problematic values.
                    4. Duplicates
                    - Report total duplicate row count.
                    - Optionally list key columns most frequently duplicated.
                    5. Outliers (ALL numeric variables)
                    - For each numeric column: detect outliers using IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
                    - Report variable name, number_of_outliers, detection_method.
                    - Explicitly show that all numeric variables were checked, even if zero outliers.
                    6. Correlation Analysis
                    - Correlation with Target (if provided): report correlation values between independent variables and the target.
                    - Correlation between independent variables: compute pairwise correlations for ALL numeric variables.
                    - List pairs with |correlation| ≥ 0.8 as high correlation pairs.
                    - Also report variables that are near-constant or redundant.
                    Use the function execute_data_exploring_plan(plan, context_variables) for every small computation step with the Coder agent, 
                    and when the analysis is complete, return the consolidated issues to the Data Engineer 
                    by calling complete_data_explore_task(results, context_variables).
            """,
            functions=[execute_data_exploring_plan, complete_data_explore_task]
        )

