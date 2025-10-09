from typing import Annotated
from pydantic import BaseModel, Field
from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

class DataEngineerStep(BaseModel):
    step_description: str = Field(
        ...,
        description=(
            "Type of data engineering step to perform."
        ),
        examples=[
            "Data cleaning",
            "Feature selection"
            "Feature engineering",
            "Data transformation",
            "Data normalization",
            "Fill missing values",
            "Encoding categorical variables"
        ]
    )
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
            # CLEANING
            "Perform data cleaning using pandas only: drop duplicates; fix dtypes; standardize category values "
            "(e.g., casing/whitespace); unify date formats; remove impossible values (e.g., negative ages) and obvious data-entry errors.",

            "Fill missing values: for numeric columns fill with median; for categorical columns fill with mode. "
            "Document the imputation strategy per column and produce a summary table of missingness before/after.",

            "Detect and treat outliers using the IQR rule for numeric features (winsorize or cap at [Q1-1.5*IQR, Q3+1.5*IQR]). "
            "Record how many values were capped per column.",

            # FEATURE SELECTION 
            "Run filter-based feature selection: remove constant/near-zero variance features; drop features with excessive "
            "missingness (e.g., >40%) or extreme cardinality where inappropriate for the task.",

            "For feature selection, compute univariate statistics: Pearson correlation (regression), ANOVA F-test "
            "(continuous vs categorical), or Chi-square (categorical vs categorical). Rank features and keep the top-K most relevant to the target variable.",

            "Remove multicollinearity : compute a suitable relationship matrix; for pairs of variables with high relationship strength, "
            "keep one representative feature based on relevance to the target and business interpretability.",

            # OUTPUTS (files & reports)
            "MANDATORY: Save the cleaned splits to the EXACT paths under ./data/ — ./data/train_cleaned.csv, ./data/val_cleaned.csv, ./data/test_cleaned.csv."
            "OVERWRITE files unconditionally (mode='w') and write CSVs with index=False. Any deviation from these filenames or location is NOT allowed."

            # IMPORTANT CONSTRAINTS
            "All cleaning and feature selection must be done "
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

                Feature Selection & Engineering (MANDATORY):
                1) Filter-based pruning
                - Remove constant/near-zero variance features.
                - Drop features with excessive missingness (e.g., >40%) or extreme cardinality when inappropriate for the task domain.
                2) Univariate relevance scoring (choose method by {problem_type} and data type)
                - Regression target: Pearson correlation |r| and/or f_regression scores; rank features and keep top-K (use provided K; otherwise default K=10).
                - Classification target: ANOVA F-test (f_classif) for continuous predictors vs class label; Chi-square for non-negative categorical counts.
                    * If using Chi-square, ensure inputs are non-negative (apply suitable counting beforehand).
                3) Multicollinearity control
                - Compute suitable relationship matrix candidate set.
                - For pairs with high relationship stength keep one representative feature based on higher univariate relevance and business interpretability; drop the rest.
                
                Data Splitting (MANDATORY):
                - Partition data into training, validation, and test sets to prevent data leakage.
                
                Workflow:
                1. Ingest Data & Review Findings:
                Begin with the raw or explored dataset, using insights from the DataExplorer or data quality reports.
                2. Execute Data Engineering Steps:
                For each cleaning or feature engineering step, call execute_data_engineer_step to delegate implementation to the Coder agent.
                3. Validation & Completion:
                - Verify that all cleaned and engineered data meets consistency and reproducibility standards.
                - Ensure that the data set will contain all the nesscessary feature to support the modeling tasks defined by the DataScientist.
                - You only be able to give the final dataset to the DataScientist and you can not ask DataScientist any things.
                - Once all preprocessing and feature engineering are complete, you have to call complete_data_engineer_task to hand off to the DataScientist.

                Rules:
                You must to make sure that no dubplicate, no missing values, no inconsistencies, and no obvious errors remain in the final datasets.
                You are not allowed to do any data encoding tasks or data normalization tasks.
                You are not allowed to do any data analysis tasks.
                Do not perform anything related to model training, evaluation, or selection. 
                Your focus ends with no issues and high-quality datasets.
                """,
            functions=[execute_data_engineer_step]
        )