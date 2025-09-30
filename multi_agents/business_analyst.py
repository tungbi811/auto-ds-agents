from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from autogen.agentchat.group import ReplyResult, AgentNameTarget, RevertToUserTarget, ContextVariables
from typing import Annotated, Literal, List, Optional
import pandas as pd

class BizAnalystOutput(BaseModel):
    business_use_case_description: str = Field(
        ...,
        description=(
            "A clear explanation of the goal of this project for the business. "
            "It should describe the problem being solved, why it matters, and "
            "how it aligns with business strategy."
        ),
        example=(
            "The goal of this project is to implement a predictive model to "
            "forecast customer churn, enabling proactive retention strategies "
            "and reducing revenue loss."
        )
    )
    business_objectives: str = Field(
        ...,
        description=(
            "Describe the impact of accurate or incorrect results on the business. "
            "Explain the benefits of high accuracy and the risks of errors."
        ),
        example=(
            "Accurate churn predictions will allow targeted retention campaigns, "
            "potentially reducing churn by 15% and saving $2M annually. "
            "Incorrect results may waste marketing spend or harm customer trust."
        )
    )
    stakeholders_expectations_explanations: str = Field(
        ...,
        description=(
            "Explain how the results will be used, who will use them, and who will "
            "be impacted by them. Identify both direct users and downstream stakeholders."
        ),
        example=(
            "The marketing team will use the predictions to design retention campaigns. "
            "Customer success managers will use them to prioritize outreach. "
            "Customers may experience more relevant engagement, improving satisfaction."
        )
    )
    problem_type: Literal["classification", "regression", "clustering"] = Field(
        ...,
        description="The type of machine learning problem to be solved.",
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
    print(df.head(5))
    return ReplyResult(
        message=f"Here is the preview:\n {df.head(5)}", 
        target=AgentNameTarget("BusinessAnalyst")
    )

def complete_business_analyst(
    output: BizAnalystOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    context_variables["business_use_case_description"] = output.business_use_case_description
    context_variables["business_objectives"] = output.business_objectives
    context_variables["stakeholders_expectations_explanations"] = output.stakeholders_expectations_explanations
    context_variables["problem_type"] = output.problem_type
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message="I have finished business understanding",
        target=AgentNameTarget("DataExplorer"),
        context_variables=context_variables,
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
                3) Produce BizAnalystOutput (business use case, objectives, stakeholders, problem type).
                4) When you finish call function `complete_business_analyst` and route to DataExplorer.
                Rules:
                - You must call functions provided: `request_clarification`, `get_data_info`, and `complete_business_analyst`.
                - Be concise, avoid jargon. No full table dumps or full-file reads.
            """,
            functions = [get_data_info, request_clarification, complete_business_analyst]
        )