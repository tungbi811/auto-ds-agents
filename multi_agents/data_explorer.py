from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class MissingValueInfo(BaseModel):
    column: str = Field(..., description="Column name with missing values.")
    missing_count: int = Field(..., description="Count of missing values in the column.")

class HighCorrelationPair(BaseModel):
    column_1: str = Field(..., description="First column in the correlated pair.")
    column_2: str = Field(..., description="Second column in the correlated pair.")
    correlation: float = Field(..., description="Correlation coefficient between the two columns.")

class DataExplorerOutput(BaseModel):
    duplicate_rows: int = Field(
        ..., description="Total number of duplicate rows in the dataset."
    )
    missing_values: List[MissingValueInfo] = Field(
        ..., description="List of columns with missing values and their details."
    )
    # target_variable: Optional[str] = Field(
    #     default=None, description="The target variable."
    # )
    # high_cardinality_cols: List[str] = Field(
    #     ..., description="List of categorical columns with unusually high unique values."
    # )
    # high_correlation_pairs: List[HighCorrelationPair] = Field(
    #     ..., description="List of feature pairs with high correlation (above 0.8 or below -0.8)."
    # )
    # constant_cols: List[str] = Field(
    #     ..., description="List of columns with only a single unique value (uninformative)."
    # )

def execute_data_exploring_plan(
    plan: Annotated[str, "One specific exploration step to implement in code (small and focused)."],
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
    return ReplyResult(
        message=f"Data exploration is complete. Here is the findings: {results.model_dump_json()}",
        target=AgentNameTarget("DataEngineer"),
        # context_variables=context_variables,
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

