from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from autogen.agentchat.group import ReplyResult, AgentNameTarget, RevertToUserTarget
from typing import Annotated, Literal, List
import pandas as pd

class BizAnalystOutput(BaseModel):
    intent_summary: str = Field(
        ..., description="1–2 sentences capturing the core business goal/question."
    )
    problem_type: Literal[
        "Regression", "Classification", "Clustering",
        "Forecasting", "Recommendation", "Segmentation", "Other"
    ] = Field(
        ..., description="ML task framing that best matches the goal."
    )
    key_metrics: List[str] = Field(
        ..., description="Business KPIs and/or ML metrics used to judge success."
    )

def request_clarification(
    clarification_question: Annotated[str, "Question to ask user for clarification"],
    # context_variables: ContextVariables,
) -> ReplyResult:
    """
    Request clarification from the user when the query is ambiguous
    """
    return ReplyResult(
        message=f"Further clarification is required to determine the correct domain: {clarification_question}",
        # context_variables=context_variables,
        target=RevertToUserTarget(),
    )

def get_data_info(
    data_path: Annotated[str, "Dataset path provided by user"],
) -> ReplyResult:
    df = pd.read_csv(data_path)
    return ReplyResult(message=f"Here is the information of columns in dataset: {df}", target=AgentNameTarget("BusinessAnalyst"))

class BusinessAnalyst(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="BusinessAnalyst",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                response_format=BizAnalystOutput,
                parallel_tool_calls=False
            ),
            system_message="""
                You are the BusinessAnalyst. 
                - Act as a senior business analyst for data science projects.
                - Bridge business needs with data capabilities.
                - Translate business requirements into clear technical requirements for the data team.
                - On every new user task, FIRST call `get_data_info` once to obtain column/data info relevant to the task.
                - ONLY if you cannot confidently fill critical fields (e.g., user_question) after step 1,
                call `request_clarification` with a concise list of targeted questions. Do not ask broad or generic questions.
                - Using the tool result + user message, produce a structured answer that conforms to `BizAnalystOutput`.
            """,
            functions = [get_data_info]
        )