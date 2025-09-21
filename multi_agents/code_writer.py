from autogen import AssistantAgent, LLMConfig
from utils.utils import execute_code

class CodeWriter(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4o-mini",
            timeout = 120,
            stream = False,
            parallel_tool_calls=False
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
                If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
                When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
            """,
            functions = [execute_code]
        )