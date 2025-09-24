from autogen import AssistantAgent, LLMConfig
from typing import Annotated
from pydantic import BaseModel, Field
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult

def execute_data_engineering_plan(
    plan: Annotated[str, "What do you want coder write code for you just one small step don't plan too much"],
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Send a small, concrete data engineering plan to the Coder for implementation.

    This function is used by the DataEngineer agent to request that the Coder 
    writes and executes code for one specific step in the data engineering process 
    (e.g., handling missing values, removing duplicates, encoding categorical variables).
    
    Args:
        plan (str): A short description of the single step of data engineering 
            to be performed. Keep the plan minimal and focused (do not include 
            multiple steps at once).
        context_variables (ContextVariables): Shared context across agents. 
            This function will update the `current_agent` field to "DataEngineer" 
            before sending the request.

    Returns:
        ReplyResult: A reply object containing the message (plan request), 
        the target agent (Coder), and the updated context variables.
    """
    context_variables["current_agent"] = "DataEngineer"
    return ReplyResult(
        message=f"Can you write code for me to execute this plan {plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables
    )

class DataEngineer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="DataEngineer",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                parallel_tool_calls=False
            ),
            system_message="""
                You are a Senior Data Engineer with deep expertise in real estate data pipelines and preprocessing.  
                Your responsibility is to read and process ALL issues identified by the Data Explorer agent, 
                then split the dataset into training and testing sets, apply full preprocessing, 
                and finally save the clean data for downstream modeling.

                Follow this process strictly:

                1. Input Understanding
                - Read the structured output from the Data Explorer, which contains all detected data problems 
                    (missing values, duplicates, unidentified values, outliers, distinct identifiers, 
                    near-constant variables, correlation problems, etc.).

                2. Data Splitting
                - Split the dataset into X_train, X_test, y_train, y_test. 
                - Ensure stratification if the target is categorical.
                - Maintain reproducibility with a fixed random_state.

                3. Problem Handling (apply fixes step by step, based on Data Explorer output)
                - Missing Values: impute or handle appropriately depending on variable type.
                - Duplicates: remove duplicate records safely.
                - Unidentified Values: clean or standardize ambiguous tokens, cast to correct type.
                - Outliers: cap, transform, or mark them depending on context.
                - Distinct Identifier Variables: drop ID-like columns that provide no predictive power.
                - Near-Constant Variables: remove variables with very low variance.
                - Correlation Problems: detect highly correlated pairs and drop redundant predictors.

                4. Standard Preprocessing
                - Encode categorical variables (OneHotEncoder, OrdinalEncoder, or similar).
                - Encode datetime features into meaningful components (year, month, day, cyclical encodings).
                - Transform text variables appropriately (drop, tokenize, or vectorize if required).
                - Normalize/scale ALL variables after encoding (numeric + encoded categorical).
                - Always fit preprocessing steps on X_train, and apply the same transform to X_test.

                5. Output
                - Save the final preprocessed X_train, X_test, y_train, y_test into the folder: data/house_price.
                - Return a summary of preprocessing actions applied 
                    (e.g., which variables were dropped, encoders used, normalization applied).
                - Ensure the final datasets are consistent, clean, and ready for modeling.

                Use the function execute_data_engineering_plan for each preprocessing step with the Coder agent.  
                Once preprocessing and saving are complete, return the final confirmation and summary 
                to downstream agents.
                """
,
            functions=[execute_data_engineering_plan]
        )
