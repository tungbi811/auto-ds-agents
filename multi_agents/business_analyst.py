from autogen import LLMConfig, AssistantAgent
from pydantic import BaseModel, Field
from typing import List, Optional
from utils.utils import request_clarification

class BizAnalystOutput(BaseModel):
    business_use_case_description: str = Field(
        description="Explain clearly what is the goal of this project for the business."
    )
    business_objectives: List[str] = Field(
        description="List the concrete objectives (what success looks like in measurable terms)."
    )
    impact_of_results: str = Field(
        description="What will be the impact of accurate or incorrect results?"
    )
    stakeholders: List[str] = Field(
        description="List of stakeholders or user groups who will use or be impacted by predictions."
    )
    stakeholder_expectations: Optional[str] = Field(
        None,
        description="Explanation of how results will be used and what stakeholders expect."
    )
    # kpis: Optional[List[str]] = Field(
    #     None, description="Key Performance Indicators that will measure success."
    # )
    # constraints: Optional[List[str]] = Field(
    #     None, description="Known constraints: data availability, budget, compliance, timeline."
    # )
    # notes: Optional[str] = Field(
    #     None, description="Any additional narrative notes."
    # )
# the output of the BusinessAnalyst must contain user input as well
class BusinessAnalyst(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="BusinessAnalyst",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                response_format=BizAnalystOutput,
                tool_choice="auto"
            ),
            system_message="""
                You are the Business Analyst Agent for the CRISP-DM Business Understanding phase.
                Your task is to take the user’s request and return a JSON object that strictly follows the BizAnalystOutput schema.
                - Always return only valid JSON that matches the schema exactly.
                - If the user does not provide enough information, use the request_clarification tool to ask clear, targeted a follow-up question until you can complete the required output (only one question at a time). Once sufficient details are gathered, return the structured JSON.
                - Keep answers clear, concise, and focused on business value.
                - Do not propose algorithms or implementation details.]
                - here is the metadata for reference
            """,
            functions = [request_clarification]
        )