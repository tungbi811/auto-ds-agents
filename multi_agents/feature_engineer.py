from autogen import AssistantAgent, LLMConfig
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union

class FeatureEngineeringStep(BaseModel):
    step_description: str = Field(
        ...,
        description="Feature engineering step to perform.",
        examples=[
            "Create ratio features",
            "Temporal features",
            "Interaction features",
            "Encode categorical variables"
        ]
    )
    action: str = Field(
        ...,
        description="Action to take for this step.",
        examples=[
            "Add debt_to_income_ratio = debt / income.",
            "Extract month and day_of_week from 'transaction_date'.",
            "Create interaction term between 'age' and 'income'.",
            "One-hot encode 'city' column."
        ]
    )
    suggestion: str = Field(
        ...,
        description="How to perform this step.",
    )
    reason: str = Field(
        ...,
        description="Reason for this step.",
        examples=[
            "Important financial indicator for risk modeling.",
            "apture seasonal or weekly spending patterns.",
            "Younger high-income vs older high-income may have different behaviors.",
            "Converts categories to numeric features usable by models."
        ]
    )
def execute_feature_engineering_step(
    step: FeatureEngineeringStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
        Delegate a single preprocessing step to the Coder agent.
        Example step: 'Impute numeric columns with median and categorical with most frequent.'
    """
    context_variables["current_agent"] = "FeatureEngineer"
    return ReplyResult(
        message=f"Please write Python code to execute this feature engineering step:\n{step}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_feature_engineering_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Feature Engineering is complete.",
        target=AgentNameTarget("Modeler"),
        context_variables=context_variables,
    )

class FeatureEngineer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="FeatureEngineer",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                parallel_tool_calls=False,
                temperature=0.3,
            ),
            system_message="""
                You are the FeatureEngineer.
                Your role is to design, create, and transform features from the cleaned datasets to maximize their utility for analysis and modeling. 
                You enhance the data by generating meaningful representations that capture underlying patterns and relationships.

                Key Responsibilities:
                - Feature creation: Derive new features from existing variables (e.g., ratios, interactions, aggregations, time-based features).
                - Feature transformation: Apply transformations such as normalization, scaling, binning, or encoding to make features more suitable for modeling.
                - Feature selection: Identify and retain features that add predictive or explanatory value, while reducing redundancy and noise.
                - Encoding categorical data: Convert categorical variables into numerical representations (e.g., one-hot, target encoding, embeddings).
                - Temporal and sequential features: Engineer lag variables, rolling statistics, or trend-based features from time-series data.
                - Domain-driven enrichment: Incorporate domain knowledge to design features that align with business logic or problem context.
                - Validation: Ensure that engineered features are consistent, non-leaky, and aligned with data quality standards.
                - Split the data into training, validation, and test sets to prevent data leakage.

                Workflow:
                1. Review the cleaned datasets and the DataCleaner’s outputs.
                2. For each feature engineering step, call execute_feature_engineering_step to delegate implementation to the Coder agent.
                3. Once all required feature engineering is complete, you must call complete_feature_engineering_task to hand off to the Modeler.

                Rules:
                - Do not perform model training, evaluation, or selection. Your scope is limited to feature engineering.
                - Ensure reproducibility and clear documentation of all engineered features.
            """,
            functions=[execute_feature_engineering_step, complete_feature_engineering_task]
        )
