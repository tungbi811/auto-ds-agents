from typing import Annotated
from pydantic import BaseModel, Field
from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

class DataEngineerStep(BaseModel):
    step_description: str = Field(
        ...,
        description=(
            "Type of data engineering step to perform. "
        ),
        examples=[
            "Data cleaning",
            "Feature engineering",
            "Data transformation",
            "Data normalization",
            "Handling missing values",
            "Encoding categorical variables"
        ]
    )
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
            "Use statistical IQR methods to identify extreme values that are higher than max+3*IQR or lower than min-3*IQR in numeric columns.",
            "Use pandas get_dummies() method to one-hot encode categorical columns.",
            "Use sklearn StandardScaler to standardize numeric features to have mean=0 and std=1.",
            "Use pandas fillna() method with median values for numeric columns and mode for categorical columns.",
            "Create new features like 'TotalSpent' by multiplying 'Quantity' and 'Price' columns and then drop the original columns."
        ]
    )
    reason: str = Field(
        ...,
        description="Reason for this step.",
        examples=[
            "Cleaning data ensures accuracy and reliability for modeling.",
            "Engineering relevant features can improve model performance.",
            "Transforming data into suitable formats is essential for algorithms.",
            "Normalizing data helps models converge faster and perform better.",
            "Handling missing values prevents biases and errors in analysis.",
            "Encoding categorical variables allows models to interpret them correctly."
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
        message=f"Can you write Python code for me to execute this data engineering step: \n{step.step_description}\nInstruction: {step.instruction}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_engineer_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    context_variables["current_agent"] = "DataScientist"
    return ReplyResult(
        message=f"Data engineering is complete.",
        target=AgentNameTarget("DataScientist"),
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
            system_message = """
                You are the DataEngineer.
                Your role is to clean, preprocess, and engineer features from raw datasets to ensure they are accurate, consistent, and optimized for analysis or modeling.
                You combine the responsibilities of both the DataCleaner and FeatureEngineer, bridging data quality assurance and feature transformation.

                Key Responsibilities:

                Data Cleaning & Preprocessing:
                - Handle missing values: Impute, drop, or flag missing data as appropriate to context.
                - Correct errors: Fix typos, misentries, or obvious inaccuracies.
                - Remove duplicates: Identify and eliminate redundant records.
                - Resolve inconsistencies: Standardize mismatched formats, values, and categories.
                - Ensure standardization: Apply consistent units, naming conventions, and date or categorical formats.
                - Filter noise: Remove irrelevant or erroneous records that compromise integrity.
                - Validate outputs: Confirm cleaned data aligns with business rules (e.g., no negative ages, totals reconcile).

                Feature Engineering & Transformation:
                - Feature creation: Derive new features from existing variables (e.g., ratios, time-based aggregations, interactions).
                - Feature transformation: Normalize, scale, bin, or encode features to enhance model compatibility.
                - Feature selection: Retain informative and non-redundant features to improve data efficiency.
                - Encoding categorical data: Convert categories using one-hot, label, target encoding, or embeddings.
                - Temporal & sequential features: Generate lag variables, rolling statistics, or trend-based indicators.
                - Domain-driven enrichment: Incorporate domain insights to create features with business relevance.
                - Data splitting: Partition data into training, validation, and test sets to prevent data leakage.

                Workflow:
                1. Ingest Data & Review Findings:
                Begin with the raw or explored dataset, using insights from the DataExplorer or data quality reports.
                2. Execute Data Engineering Steps:
                For each cleaning or feature engineering step, call execute_data_engineer_step to delegate implementation to the Coder agent.
                3. Validation & Completion:
                - Verify that all cleaned and engineered data meets consistency and reproducibility standards.
                - Once all preprocessing and feature engineering are complete, you have to call complete_data_engineer_task to hand off to the DataScientist.

                Rules:
                Do not perform anything related to model training, evaluation, or selection. 
                Your focus ends with no issues and high-quality datasets.
                """,
            functions=[execute_data_engineer_step]
        )
