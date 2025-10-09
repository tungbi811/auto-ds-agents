from typing import Annotated
from pydantic import BaseModel, Field
from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

class DataEngineerStep(BaseModel):
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
            "Remove duplicate rows based on all columns except the index or unique identifier.",
            "Standardize date formats to YYYY-MM-DD across all date columns.",
            "Impute missing values in numeric columns using median and categorical columns using mode.",
            "Use sklearn StandardScaler to standardize numeric features to have mean=0 and std=1.",
            "For categorical variables with high cardinality, use frequency encoding.",
            "Remove columns that have more than 30% missing values.",
            "Create new features like 'TotalSpent' by multiplying 'Quantity' and 'Price' columns and then drop the original columns."
        ]
    )

def execute_data_engineer_step(
    step: DataEngineerStep,
    context_variables: ContextVariables
) -> ReplyResult:
    """
        Delegate coding of a specific data engineering step to the Coder agent.
    """
    context_variables["current_agent"] = "DataEngineer"
    return ReplyResult(
        message=f"Hey Coder! Here is the instruction for the data engineering step: \n{step.instruction} \nCan you write Python code for me to execute it?",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

class DataEngineer(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type="openai",
            model="gpt-4.1-mini",
            temperature=0.1,
            stream=False,
            parallel_tool_calls=False
        )
        
        super().__init__(
            name="DataEngineer",
            llm_config=llm_config,
            system_message = """
                You are the DataEngineer.
                Your role is to clean, preprocess, and engineer features from raw datasets to ensure they are accurate, consistent, and optimized for analysis or modeling. 
                You transform raw data into structured, high-quality, and model-ready datasets.

                Workflow:
                1. Load and Inspect Data
                - Ingest the raw dataset and review its structure, data types, and completeness.
                - Identify potential data quality issues such as missing values, inconsistencies, and duplicates.

                2. Data Cleaning
                - Handle missing or invalid values appropriately (imputation, removal, or flagging).
                - Correct data errors and inconsistencies.
                - Remove duplicate and irrelevant records.
                - Standardize column formats, naming conventions, and categorical values.
                - Validate that cleaned data satisfies logical and business rules (e.g., no negative ages, valid dates).

                3. Feature Engineering
                - Create new meaningful features (e.g., ratios, aggregations, time-based or domain-specific features).
                - Transform existing variables through scaling, normalization, binning, or encoding as needed.
                - Select and retain features that add analytical or predictive value.
                - Do not perform dimensionality reduction like PCA.

                4. Execution of Steps
                - For each cleaning or feature engineering task, call execute_data_engineer_step to delegate implementation to the Coder agent.

                5. Validation and Summarization
                - Confirm that the final dataset meets data quality and consistency standards.
                - Summarize all key cleaning and feature engineering actions, including the final dataset’s shape, key transformations, and quality checks.

                Rules:
                - Do not perform dimensionality reduction (e.g., PCA), model training, evaluation, or selection.
                - Focus only on data cleaning and feature engineering.
                - Ensure outputs are clean, standardized, and fully reproducible.
                - Maintain clear documentation of all transformations performed.
                """
,
            functions=[execute_data_engineer_step]
        )
