from autogen import AssistantAgent, LLMConfig
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union


class DataCleaningPlan(BaseModel):
    plan_type: Literal[
        "Remove column that has more than 50% missing values",
    ]
    columns: Optional[List[str]] = Field(

class DataEngineerOutput(BaseModel):
    X_train_path: str = Field(..., description="Path to training features CSV.")
    y_train_path: Optional[str] = Field(
        None, description="Path to training target CSV if supervised; None for unsupervised tasks."
    )
    X_val_path: Optional[str] = Field(
        None, description="Path to validation features CSV if available; None otherwise."
    )
    y_val_path: Optional[str] = Field(
        None, description="Path to validation target CSV if available and supervised; None otherwise."
    )
    X_test_path: str = Field(..., description="Path to test features CSV if available.")
    y_test_path: Optional[str] = Field(
        None, description="Path to test target CSV if available and supervised; None otherwise."
    )

class DataCleaningPlan(BaseModel):
    plan_type: Literal[
        "handle_missing_values", "remove_duplicates", "handle_outliers", "standardize_formats", "validate_data"
    ]
    plan_description: str = Field(
        ...,
        description="Detailed explanation of the specific step in the data cleaning process.",
        example="Impute missing values in 'age' column with median."
    )

def execute_data_cleaning_plan(
    plan: DataCleaningPlan
):
    pass

def execute_feature_engineering_plan():
    pass

def execute_data_preparation_plan():
    pass

def execute_data_engineering_plan(
    plan: DataEngineeringPlan,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
        Delegate a single preprocessing step to the Coder agent.
        Example plan: 'Impute numeric columns with median and categorical with most frequent.'
    """
    context_variables["current_agent"] = "DataEngineer"
    return ReplyResult(
        message=f"Please write Python code to execute this data engineering step:\n{plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_engineering(
    output: DataEngineerOutput,
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Data engineering is complete. Here are the dataset paths: {output.json()}",
        target=AgentNameTarget("Modeler"),
        context_variables=context_variables,
    )

class DataEngineer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="DataEngineer",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                parallel_tool_calls=False,
                response_format=DataEngineerOutput,
                temperature=0.3,
            ),
            system_message="""
                You are the DataEngineer.
                - Your role is to clean and preprocess data based on the DataExplorer's findings and the BizAnalyst's goal.
                Workflow:
                1) Use `execute_data_cleaning_plan` to clean data based on identified issues base on DataExplorer's finding issue.
                2) Use `execute_feature_engineering_plan` to create or transform features as needed

                - Use `execute_data_engineering_plan` to delegate coding of specific preprocessing steps to the Coder agent.
                - When all necessary preprocessing is done, call `complete_data_engineering` with paths to final datasets.
                - Ensure the data is ready for modeling, with no missing values or unhandled issues.
                - When saving dataset, tell the coder to use X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv 
                as file names if exists and save to the same folder with the original dataset.
            """,
            functions=[execute_data_engineering_plan, complete_data_engineering]
        )
