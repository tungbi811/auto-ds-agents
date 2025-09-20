from autogen import AssistantAgent, LLMConfig
from pydantic import BaseModel, Field
from typing import List

class SummariserOutput(BaseModel):
    summary: str = Field(..., description="Concise narrative of what happened.")
    key_findings: List[str] = Field(
        default_factory=list, description="Notable results, metrics, or observations."
    )
    issues: List[str] = Field(
        default_factory=list, description="Problems or anomalies detected (if any)."
    )

class Summariser(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4o-mini",
            timeout = 120,
            stream = False,
            response_format=SummariserOutput,
            parallel_tool_calls=False
        )
        super().__init__(
            name="Summariser",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Summariser. Given execution results, produce a concise summary of the key findings, metrics, 
                and any issues observed. Do not judge success or failure—only highlight what happened, what outputs were 
                produced, and any anomalies. Pass your summary to the Evaluator.
            """
        )