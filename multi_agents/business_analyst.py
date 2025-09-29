from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from autogen.agentchat.group import ReplyResult, AgentNameTarget, RevertToUserTarget, ContextVariables
from typing import Annotated, Literal, List, Optional
import pandas as pd

class BizAnalystOutput(BaseModel):
    goal: str = Field(
        ..., description="1–2 sentences capturing the core business goal/question."
    )
    problem_type: Literal[
        "Regression", "Classification", "Clustering",
        "Forecasting", "Recommendation", "Anomaly Detection"
    ] = Field(
        ..., description="ML task framing that best matches the goal."
    )

def request_clarification(
    clarification_question: Annotated[str, "One targeted question to clarify user intent"],
) -> ReplyResult:
    """
    Request clarification from the user when the query is ambiguous
    """
    return ReplyResult(
        message=f"Further clarification is required to determine the correct domain: {clarification_question}",
        target=RevertToUserTarget(),
    )

def get_data_info(
    data_path: Annotated[str, "Dataset path"],
) -> ReplyResult:
    df = pd.read_csv(data_path)
    print(df.head())
    return ReplyResult(message=f"Here is the information of columns in dataset", target=AgentNameTarget("BusinessAnalyst"))

def complete_business_analyst(
    output: BizAnalystOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    context_variables["goal"] = output.goal
    context_variables["problem_type"] = output.problem_type

    return ReplyResult(
        message="I have finished business understanding",
        target=AgentNameTarget("DataExplorer")
    )

class BusinessAnalyst(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="BusinessAnalyst",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                response_format=BizAnalystOutput,
                temperature=0.3,
                parallel_tool_calls=False
            ),
            system_message="""
                You are the BizAnalyst.
                - Translate the user’s request into a concrete DS task.
                Workflow:
                1) If the task depends on a dataset and no path is given, call `request_clarification` with ONE targeted question.
                2) If a path is given, call `get_data_info` ONCE to peek (tiny head/sample) to ground your framing.
                3) Produce BizAnalystOutput (goal, problem_type, key_metrics).
                4) When you finish call function `complete_business_analyst` and route to DataExplorer.
                Rules:
                - Be concise, avoid jargon. No full table dumps or full-file reads.
            """,
            functions = [get_data_info, request_clarification, complete_business_analyst]
        )