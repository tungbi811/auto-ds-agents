from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from autogen.agentchat.group import ReplyResult, AgentNameTarget, RevertToUserTarget, ContextVariables
from typing import Annotated, Literal, List, Optional
import pandas as pd

class BizAnalystOutput(BaseModel):
    objective: str = Field(
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
    research_questions: List[str] = Field(
        ...,
        description="A list of specific research questions that the analysis aims to answer.",
        example=[
            "What demographic or behavioral characteristics most strongly correlate with churn?",
            "What role do service-related factors (delivery delays, complaints) play in customer attrition?",
            "How does customer engagement correlate with retention rates?"
        ]
    )
    problem_type: Literal["classification", "regression", "clustering", "time_series", "anomaly_detection", "recommendation"] = Field(
        ...,
        description="The type of machine learning problem that best fits the business use case.",
        example="classification"
    )

def request_clarification(
    clarification_question: Annotated[str, "One targeted question to clarify user requirements"],
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
    context_variables["objective"] = output.objective
    context_variables["research_questions"] = output.research_questions
    context_variables["problem_type"] = output.problem_type
    context_variables["current_agent"] = "DataAnalyst"
    return ReplyResult(
        message="I have finished business understanding",
        target=AgentNameTarget("DataAnalyst"),
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
                Your role is to transform user requirements into structured, actionable business analysis outputs. 
                You ensure clarity of the business context, goals, stakeholder expectations, and the research questions 
                that guide exploration.

                Key Responsibilities:
                - Define objectives: List clear, measurable business goals that align with the use case.
                - Formulate research questions: Define the analytical questions that need to be answered to achieve the objectives.
                - Get data info: always call get_data_info first to discover what datasets, variables, and metadata are available for the project.
                - Complete business analysis: When you have a clear understanding of the business context, objectives, and research questions, you must call complete_business_analyst to hand off to the DataExplorer.
                
                Workflow:
                1. Review initial user requirements.
                2. call get_data_info first to discover what datasets, variables, and metadata are available for the project.
                3. If requirements are vague, incomplete, or conflicting, call request_clarification to ask for more details.
                4. Break requirements into structured outputs:
                - objective
                - research_questions
                - problem_type
                3. When complete, you must call complete_business_analysis_task to hand off to the DataExplorer.

                Rules:
                - Do not propose data cleaning, feature engineering, or modeling directly.
                - Keep analysis high-level, business-focused, and actionable.
            """,
            functions = [get_data_info, request_clarification, complete_business_analyst]
        )