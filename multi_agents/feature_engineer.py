from autogen import AssistantAgent, LLMConfig
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

class FeatureEngineeringPlan(BaseModel):
    step_type: Literal[
        "one_hot_encode", "label_encode", "frequency_encode", "target_encode", "correlation_feature_selection", 
        "variance_feature_selection", "scale_features", "perform_pca", "perform_rfe", "create_polynomial_features", 
        "create_feature_combinations"
    ] = Field(..., description="Type of feature engineering step to perform.")
    rule: str = Field(..., description="Specific rule or method to apply for this step.")
    columns: Optional[List[str]] = Field(
        None, description="List of columns to apply the step to. If None, apply to all relevant columns."
    )

def execute_feature_engineering_plan(
    plan: FeatureEngineeringPlan,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
        Delegate a single preprocessing step to the Coder agent.
        Example plan: 'Impute numeric columns with median and categorical with most frequent.'
    """
    context_variables["current_agent"] = "FeatureEngineer"
    return ReplyResult(
        message=f"Please write Python code to execute this feature engineering step:\n{plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_feature_engineering_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Feature Engineering is complete.",
        target=AgentNameTarget("FeatureEngineer"),
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
                temperature=0.3,
            ),
            system_message="""
                You are the DataEngineer.
                - Your role is to clean and preprocess data based on the DataExplorer's findings and the BizAnalyst's goal.
                - Use `execute_data_engineering_plan` to delegate coding of specific preprocessing steps to the Coder agent.
                - When all necessary preprocessing is done, call `complete_data_engineering` with paths to final datasets.
                - Ensure the data is ready for modeling, with no missing values or unhandled issues.
                - When saving dataset, tell the coder to use X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv 
                as file names if exists and save to the same folder with the original dataset.
            """,
            functions=[execute_feature_engineering_plan, complete_feature_engineering_task]
        )
