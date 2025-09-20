from autogen import AssistantAgent, LLMConfig
from pydantic import BaseModel, Field
from typing import List

class CodeWriter(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "google",
            model = "gemini-2.5-pro",
            timeout = 120,
            stream = False
        )
        super().__init__(
            name="CodeWriter",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Code Writer. 
                Your role is to take the Planner’s structured step-by-step instructions and write complete, runnable code. 
                Generate Python code inside a full fenced code block (python ... ).
                Ensure the code is correct, minimal, and follows best practices. 
                When execution errors occur, carefully analyze the error logs and provide a corrected version of the code. 
                Always return only the updated code in your output, without extra commentary.
                
            """
        )