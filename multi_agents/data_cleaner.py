from autogen import AssistantAgent, LLMConfig
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

class DataCleaningPlan(BaseModel):
    step_type: Literal[
        "fill_missing_values", "remove_columns_with_missing_data", "detect_and_handle_outliers_zscore", 
        "detect_and_handle_outliers_iqr", "convert_data_types", "remove_duplicates", "format_datetime"
    ] = Field(..., description="Type of data cleaning step to perform.")
    rule: str = Field(..., description="Specific rule or method to apply for this step.")
    columns: Optional[List[str]] = Field(
        None, description="List of columns to apply the step to. If None, apply to all relevant columns."
    )

def execute_data_cleaning_plan(
    plan: DataCleaningPlan,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
        Delegate a single preprocessing step to the Coder agent.
        Example plan: 'Impute numeric columns with median and categorical with most frequent.'
    """
    context_variables["current_agent"] = "DataCleaner"
    return ReplyResult(
        message=f"Please write Python code to execute this data cleaning step:\n{plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_cleaning_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Data Cleaning is complete.",
        target=AgentNameTarget("FeatureEngineer"),
        context_variables=context_variables,
    )

class DataCleaner(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="DataCleaner",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                parallel_tool_calls=False,
                temperature=0.3,
            ),
            system_message="""
                You are the DataCleaner.
                - Your role is to clean and preprocess data based on the DataExplorer's findings and the BizAnalyst's goals.
                - Use `execute_data_cleaning_plan` to delegate coding of specific preprocessing steps to the Coder agent.
                - When all necessary preprocessing is done, call `complete_data_cleaning` with paths to final datasets.
            """,
            functions=[execute_data_cleaning_plan, complete_data_cleaning_task]
        )
