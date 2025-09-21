from autogen import AssistantAgent, LLMConfig
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class EvaluatorFeedback(BaseModel):
    what: str = Field(..., description="What went wrong.")
    where: str = Field(..., description="Where it occurred (step, code, data, etc.).")
    why: str = Field(..., description="Why it failed.")

class EvaluatorResults(BaseModel):
    artifacts: List[str] = Field(
        default_factory=list,
        description="Links/paths/identifiers of final outputs for the user."
    )

class EvaluatorOutput(BaseModel):
    evaluation: Literal["pass", "fail"] = Field(..., description="Did it meet success criteria?")
    justification: str = Field(..., description="Reasoning for the outcome.")
    actionable_feedback: Optional[EvaluatorFeedback] = Field(
        None,
        description="Present only when evaluation == 'fail'."
    )
    final_results: Optional[EvaluatorResults] = Field(
        None,
        description="Present only when evaluation == 'pass'."
    )

class Evaluator(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4o-mini",
            timeout = 120,
            stream = False,
            response_format=EvaluatorOutput
        )
        super().__init__(
            name="Evaluator",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Evaluator. Compare the Summariser’s output against the defined success criteria. 
                If the task passes, return the final results and artifacts to the user. If the task fails, 
                provide actionable feedback: explain what went wrong, where it occurred, and why it failed. 
                Feedback should guide the Planner and Code Writer in revising their steps and code.
            """
        )