from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal


class VariableReport(BaseModel):
    variable: str = Field(..., description="Name of the variable (column).")
    insight: str = Field(..., description="Detailed insight about this variable from EDA.")
    issue: str = Field(
        ..., 
        description="Main issue for this variable (e.g., missing values, outliers, all unique values, low correlation with target, high correlation with other variables)."
    )

class DataExplorerOutput(BaseModel):
    variables: List[VariableReport] = Field(
        ..., description="List of variables with their insights and issues."
    )


def execute_code(
    plan: Annotated[str, "What do you want coder write code for you just one small step don't plan too much"],
    context_variables: ContextVariables) -> ReplyResult:
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message=f"Can you write code for me to execute this plan {plan}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables
    )

# def complete_data_explore_task(results: DataExplorerOutput, context_variables: ContextVariables) -> ReplyResult:
#     return ReplyResult(
#         message=f"Issue: {results.missing_values}",
#         target=AgentNameTarget("DataEngneer")
#     )

class DataExplorer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                response_format=DataExplorerOutput,
                parallel_tool_calls=False
            ),
            system_message = """
                You are a Senior Data Analyst with deep expertise in real estate data and data science projects.  
                Your responsibility is to perform a structured exploratory data analysis (EDA) to ensure the dataset is well-understood, potential issues are identified, and meaningful insights are generated for downstream data engineering and modeling phases.  

                Follow this process step by step:

                1. **Dataset Understanding**  
                - Identify the target variable and its type (binary, categorical, continuous).  
                - Summarize the dataset structure (rows, columns, variable types).  
                - Detect identifier variables (e.g., IDs, unique keys) that are not suitable for modeling.  

                2. **Data Quality Diagnostics**  
                - Missing Values: detect variables with missing data, count missing values, calculate missing rate.  
                - Duplicates: identify number of duplicate records and columns where duplication occurs.  
                - Unidentified Values: detect variables containing values that cannot be properly typed (e.g., mixed formats, ambiguous data).  

                3. **Outlier Analysis**  
                - Detect and quantify extreme values.  
                - Classify them into harmful outliers (errors, noise) vs meaningful outliers (rare but important business cases).  
                - Summarize their potential impact on analysis and modeling.  

                4. **Correlation & Relationships**  
                - Measure correlation of independent variables with the target variable.  
                - Detect strong correlations between independent variables (multicollinearity).  
                - Highlight weak or redundant predictors.  

                5. **Distribution & Patterns**  
                - Provide summary of variable distributions (categorical frequencies, numerical histograms, etc.).  
                - Highlight skewness, imbalance, or unusual trends relevant to real estate data.  

                6. **Problem Identification**  
                - Clearly list variables and issues under each category (missing values, outliers, duplicates, correlations, etc.).  
                - Do not propose solutions, only describe the problems and their potential impact.  

            """,
            functions=[execute_code]
        )

