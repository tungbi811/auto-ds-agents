from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class DataExploringStep(BaseModel):
    step_description: str= Field(
        ...,
        description=(
            "Type of exploration step to perform. "
        ),
        examples=[
            "Profile dataset",
            "Missing value analysis",
            "Outlier detection",
            "Duplicate check.",
            "Inconsistency check",
        ]
    )

    action: str = Field(
        ...,
        description="Action to take for this step.",
        examples=[
            " Summarize dataset shape, column names, data types, and number of unique values per column.",
            "Calculate the percentage of missing values per column and detect patterns (e.g., missing at random vs systematic).",
            "Use statistical methods (e.g., z-score, IQR) to identify extreme values in numeric columns.",
            "Identify duplicate rows (exclude id columns) and duplicate keys (e.g., user_id).",
            "Detect inconsistent formats (e.g., date formats, category labels like 'Male' vs 'M')."
        ]
    )

    reason: str = Field(
        ...,
        description="Reason for this step.",
        examples=[
            "Provides a high-level understanding of the dataset structure.",
            "Identifies potential risks and informs cleaning strategies.",
            "Detects anomalies that could distort analyses or models.",
            "Ensures uniqueness and integrity of records.",
            "Highlights data standardization issues."
        ]
    )

class DataExplorerOutput(BaseModel):
    issues: List[str] = Field(
        ...,
        description="A list of data quality issues identified during data exploration."
    )

    insights: List[str] = Field(
        ...,
        description="A list of key insights discovered during data exploration."
    )

def execute_data_exploring_step(
    step: DataExploringStep,
    context_variables: ContextVariables
) -> ReplyResult:
    """
        Delegate coding of a specific exploration step to the Coder agent.
    """
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message=f"Can you write Python code for me to execute this exploration step: \n{step}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_exploring_task(
    results: DataExplorerOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Complete the DataExplorer stage and hand off results to the DataEngineer.
    """
    context_variables["data_insights"] = results.insights
    context_variables["data_issues"] = results.issues
    context_variables["current_agent"] = "DataCleaner"
    return ReplyResult(
        message=f"Here are the data quality insights {results.insights} and issues found: {results.issues}",
        target=AgentNameTarget("DataCleaner"),
    )

class DataExplorer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                response_format=DataExplorerOutput,
                parallel_tool_calls=False
            ),
            system_message = """
                You are the DataExplorer.
                Your role is to explore datasets in order to discover insights, patterns, and issues. You provide a clear picture 
                of the data’s structure, quality, and behavior, so that downstream agents (DataCleaner and FeatureEngineer) can act effectively.

                Key Responsibilities:
                - Profile the dataset: Summarize structure, column types, distributions, and ranges.
                - Detect missing values: Report the extent and patterns of missing data.
                - Identify outliers and anomalies: Highlight unusual or extreme values.
                - Spot errors and inconsistencies: Note typos, misentries, formatting problems, or mismatched categories.
                - Check duplicates: Report duplicate rows or entities.
                - Generate descriptive statistics: Provide summaries (mean, median, mode, variance, correlations, etc.).
                - Surface potential insights: Highlight trends, patterns, or relationships that stand out.

                Workflow:
                1. Review the dataset to identify areas for exploration.
                2. For each exploration step, call execute_data_exploring_step to delegate implementation to the Coder agent.
                3. When exploration is complete, summarise it into structured output and call complete_data_exploring_task.
                
                Rules:
                Do not clean, transform, or engineer features — only explore and report.
                Do not perform model training or evaluation.
                Findings should be clear, concise, and useful for the next steps.
            """,
            functions=[execute_data_exploring_step, complete_data_exploring_task]
        )
