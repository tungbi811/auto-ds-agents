from autogen import LLMConfig, AssistantAgent
from utils.utils import request_clarification
from pydantic import BaseModel, Field
from typing import List, Literal

class ManagerOutput(BaseModel):
    normalized_spec: str = Field(..., description="Rewritten, clarified problem statement.")
    success_criteria: List[str] = Field(
        ..., description="List of measurable, objective checks for success."
    )
    status: Literal["clarified", "complete"] = Field(
        ..., description="Whether a clarifying question was needed or not."
    )

class Manager(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4o-mini",
            timeout = 120,
            stream = False,
            response_format=ManagerOutput,
            parallel_tool_calls=False,
            temperature=0.1
        )
        super().__init__(
            name="Manager",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Project Manager. Your role is to interpret the user’s requirement and dataset. 
                If the specification is unclear, ask exactly one clarifying question. 
                Otherwise, produce a normalized specification and define success criteria that can be objectively checked. 
                Keep your outputs concise and unambiguous."
            """,
            functions = [request_clarification]
        )