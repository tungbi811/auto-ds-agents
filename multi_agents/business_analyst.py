from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from autogen.agentchat.group import ReplyResult, AgentNameTarget, RevertToUserTarget, ContextVariables
from typing import Annotated, Literal, List
import pandas as pd
from utils.utils import convert_message_to_markdown

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
    stakeholders_expectations: str = Field(
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
    markdown_response = convert_message_to_markdown(clarification_question)
    return ReplyResult(
        message=markdown_response,
        target=RevertToUserTarget(),
    )

def get_data_info(
    data_path: Annotated[str, "Dataset path"],
) -> ReplyResult:
    df = pd.read_csv(data_path)
    markdown_response = convert_message_to_markdown(f"Take a look at the first few rows of the dataset:\n {df.head(5)} \n Numerical columns: {df.select_dtypes(include=['number']).columns.tolist()} \n Categorical columns: {df.select_dtypes(include=['object', 'category']).columns.tolist()} \n Date columns: {df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()} \n Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    return ReplyResult(
        message=markdown_response,
        target=AgentNameTarget("BusinessAnalyst")
    )

def complete_business_analyst(
    output: BizAnalystOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    context_variables["objective"] = output.objective
    context_variables["research_questions"] = output.research_questions
    context_variables["problem_type"] = output.problem_type
    context_variables["stakeholders_expectations"] = output.stakeholders_expectations
    markdown_response = convert_message_to_markdown(f"""The business analysis is complete with the following details:
            - Objective: {output.objective}
            - Stakeholder Expectations: {output.stakeholders_expectations}
            - Research Questions: {', '.join(output.research_questions)}
            - Problem Type: {output.problem_type}
            """)
    return ReplyResult(
        message=markdown_response,
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
    )

class BusinessAnalyst(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type= "openai",
            model="gpt-4.1-mini",
            temperature=0.5,
            stream=False,
            parallel_tool_calls=False,
        )
        super().__init__(
            name="BusinessAnalyst",
            llm_config=llm_config,
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