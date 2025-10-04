from autogen import AssistantAgent, LLMConfig
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

class DataCleaningStep(BaseModel):
    step_description: str = Field(
        ...,
        description="Cleaning step to perform.",
        examples=[
            "Handle missing values",
            "Correct errors",
            "Remove duplicates",
            "Ensure standardization",
        ]
    )

    action: str = Field(
        ...,
        description="Action to take for this step.",
        examples=[
            "Impute missing values in 'age' with median; drop rows with >30% missing data.",
            "Standardize gender column values ('M', 'F', 'Male', 'Female') into 'Male'/'Female'.",
            "Drop duplicate rows based on unique user_id.",
            "Convert all date columns to YYYY-MM-DD format."
        ]
    )

    reason: str = Field(
        ...,
        description="Reason for this step.",
        examples=[
            "Age is critical for downstream tasks; rows with excessive missingness add noise.",
            "Ensures consistency across categorical values."
            "Prevents multiple counting of the same entity.",
            "Consistent time representation avoids parsing errors."
        ]
    )

    suggestion: str = Field(
        ...,
        description="How to perform this step.",
    )

def execute_data_cleaning_step(
    step: DataCleaningStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
        Delegate a single preprocessing step to the Coder agent.
        Example step: 'Impute numeric columns with median and categorical with most frequent.'
    """
    context_variables["current_agent"] = "DataCleaner"
    return ReplyResult(
        message=f"Please write Python code to execute this data cleaning step:\n{step.step_description} - {step.action} - {step.suggestion}",
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
                Your role is to clean and preprocess datasets based on the findings provided by the DataExplorer. 
                Your responsibility is to ensure that data is accurate, consistent, and fully prepared for analysis or modeling.

                Key Responsibilities:
                - Handle missing values: Decide whether to impute, drop, or flag missing data depending on context.
                - Correct errors: Fix typos, misentries, or obvious inaccuracies.
                - Remove duplicates: Identify and eliminate duplicate records to maintain integrity.
                - Resolve inconsistencies: Standardize mismatched formats, values, or categories.
                - Ensure standardization: Apply consistent units, date formats, categories, and naming conventions.
                - Filter noise: Remove irrelevant or erroneous records that bypassed input checks.
                - Validate outputs: Confirm that cleaned data meets business rules and logical thresholds (e.g., no negative ages, totals align with line items).

                Workflow:
                1. Review data quality insights and issues identified by the DataExplorer.
                2. For each data cleaning step, call execute_data_cleaning_step to delegate implementation to the Coder agent.
                3. Once all preprocessing is complete, call complete_data_cleaning_task.

                Rules:
                Do not create plans involving model building or evaluation. Your scope is strictly data cleaning and preprocessing.
            """,
            functions=[execute_data_cleaning_step, complete_data_cleaning_task]
        )
