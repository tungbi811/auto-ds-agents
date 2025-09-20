from autogen import AssistantAgent, LLMConfig
from pydantic import BaseModel, Field
from typing import List

class PlannerStep(BaseModel):
    step_id: int = Field(..., ge=1, description="Monotonic step number starting at 1.")
    instruction: str = Field(..., description="Clear action for Code Writer.")
    expected_output: str = Field(..., description="What should be produced by this step.")

class PlannerOutput(BaseModel):
    steps: List[PlannerStep] = Field(..., min_items=1, description="Ordered execution steps.")
    revised: bool = Field(False, description="True if this plan revises a previous failed attempt.")

class Planner(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4o-mini",
            timeout = 120,
            stream = False,
            response_format=PlannerOutput,
            parallel_tool_calls=False
        )
        super().__init__(
            name="Planner",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Planner. Given a normalized specification and success criteria, break the problem into clear, 
                ordered steps with expected outputs. Your plan must be executable by a code-writing agent, so be explicit, structured, and testable. 
                If you receive evaluator feedback, revise the plan to address the failure cause while keeping it efficient.
            """
        )