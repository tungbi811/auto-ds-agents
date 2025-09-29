from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class DataExploringPlan(BaseModel):
    """
    Defines a structured plan for exploring a dataset, focusing on specific
    data quality and statistical checks.
    """

    step_type: Literal["dtype", "duplicate_rows", "missing_values","unique_values","target_variable_analysis"] = Field(
        ...,
        description=(
            "Type of exploration step to perform. "
            
        ),
        example="missing_values"
    )

    rule: str = Field(
        ...,
        description=(
            "Specific rule, threshold, or method to apply for this step. "
            "This should clarify how the step will be executed."
        ),
        example="Report columns with missing values only"
    )

    columns: Optional[List[str]] = Field(
        None,
        description=(
            "List of specific columns to apply the step to. "
            "If None, apply the step to all relevant columns automatically."
        ),
        example=["age", "income", "gender"]
    )

class MissingValueInfo(BaseModel):
    column: str = Field(..., description="Column name with missing values.")
    missing_count: int = Field(..., description="Count of missing values in the column.")

class DataExplorerOutput(BaseModel):
    total_rows: int = Field(..., description="Total number of rows in the dataset.")
    total_columns: int = Field(..., description="Total number of columns in the dataset.")
    numerical_columns: List[str] = Field(
        ..., description="List of numerical columns in the dataset."
    )
    categorical_columns: List[str] = Field(
        ..., description="List of categorical columns in the dataset."
    )
    datetime_columns: List[str] = Field(
        ..., description="List of datetime columns in the dataset."
    )
    duplicate_rows: int = Field(
        ..., description="Total number of duplicate rows in the dataset."
    )
    missing_values: List[MissingValueInfo] = Field(
        ..., description="List of columns with missing values and their details."
    )
    target_column: Optional[str] = Field(
        default=None, description="Base on the problem statement, create or identify the target variable if needed."
    )
    target_variable_insight: Optional[Dict[str, int]] = Field(
        default=None,
        description="provide a detailed analysis on the target variable, its distribution, limitations, issues, ..."
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
) -> ReplyResult:
    """
    Complete the DataExplorer stage and hand off results to the DataEngineer.
    """
    return ReplyResult(
        message=f"Data exploration is complete. Here is the findings: {results}",
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
                - Always keep exploration steps small and focused.
                - Use `execute_data_exploring_plan` to delegate coding to the Coder agent.
                - When exploration is finished, call `complete_data_explore_task` with your findings.
                - Do not build models or perform heavy transformations. Just identify problems and opportunities.
            """,
            functions=[execute_data_exploring_plan, complete_data_explore_task]
        )

