from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class DataExploringPlan(BaseModel):
    plan_description: str = Field(
        ...,
        description=(
            "A concise description of the overall exploration plan. It should outline the key steps to take in order to understand the dataset."
        ),
        example=(
            "Explore the relationship between age, income, and spending habits in the dataset."
        )
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
    column: str = Field(
        ...,
        description="The name of the column where the issue was found.",
        example="income"
    )
    issue: str = Field(
        ...,
        description="A concise description of the data quality issue discovered.",
        example="There are 15% missing values in this column."
    )
    recommendation: str = Field(
        ...,
        description="A recommended action to address the identified issue.",
        example=(
            "Consider imputing missing values with the median or removing rows with missing data.",
            "Remove outliers using the IQR method."
            ""
        )
    )

class NexxtStep(BaseModel):
    columns: List[str] = Field(
        ...,
        description="List of columns to apply the next step to.",
        example=["age", "income"]
    )
    action: Literal[
        "impute_missing_values", "remove_outliers", "convert_data_types", "remove_duplicates", "format_datetime"
    ] = Field(
        ...,
        description="The type of data cleaning or feature engineering action to take.",
        example="impute_missing_values" 
    )

class DataExplorerOutput(BaseModel):
    insights: List[Insight] = Field(
        ...,
        description="A list of key insights discovered during data exploration."
    )
    issues: List[Issue] = Field(
        ...,
        description="A list of data quality issues identified during data exploration."
    )
    recommended_next_steps: List[str] = Field(
        ...,
        description="A list of recommended next steps for data cleaning or feature engineering based on the findings."
    )

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

def complete_data_explore_task(
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
        message=f"Here are the recommended next steps: {results.recommended_next_steps}",
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
                4) When exploration is finished, call `complete_data_explore_task` with your findings and route to Data Cleaner.
                Rules:
                - You must call two functions provided: `execute_data_exploring_plan` and `complete_data_explore_task`.
                - Don't perform coding yourself. Always delegate coding to the Coder agent.
                - Don't make plan for data cleaning, feature engineering or modeling. Focus only on exploration and summarization.
            """,
            functions=[execute_data_exploring_plan, complete_data_explore_task]
        )

