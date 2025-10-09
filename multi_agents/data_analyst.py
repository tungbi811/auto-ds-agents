from typing import List
from pydantic import BaseModel, Field
from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget

class DataAnalystStep(BaseModel):
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
            "Use statistical IQR methods to identify extreme values that are higher than max+3*IQR or lower than min-3*IQR in numeric columns.",
            "Check for duplicate rows excluding the index column or unique identifier.",
            "Calculate the number and percentage of missing values for each column.",
        ]
    )

def execute_data_analyst_step(
    step: DataAnalystStep,
    context_variables: ContextVariables
) -> ReplyResult:
    """
        Delegate coding of a specific exploration step to the Coder agent.
    """
    context_variables["current_agent"] = "DataAnalyst"
    return ReplyResult(
        message=f"Coder! Can you write Python code to :\n {step.instruction}. \n",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

class DataAnalyst(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type= "openai",
            model="gpt-4.1-mini",
            temperature=0.3,
            stream=False,
            parallel_tool_calls=False
        )
        super().__init__(
            name = "DataAnalyst",
            llm_config=llm_config,
            system_message = """
                You are the DataAnalyst.
                Your role is to explore datasets in order to discover insights, patterns, and issues. You provide a clear picture 
                of the data’s structure, quality, and behavior, so that downstream Data Engineer can act effectively.

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
                2. For each exploration step, call execute_data_analyst_step to delegate implementation to the Coder agent.
                3. When exploration is complete, summarise insights and findings.

                Rules:
                Do not clean, transform, or engineer features — only explore and report.
                Do not perform model training or evaluation.
                Findings should be clear, concise, and useful for the next steps.
            """,
            functions=[execute_data_analyst_step]
        )
