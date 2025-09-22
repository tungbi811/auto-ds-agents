from pathlib import Path
from typing import Annotated
from autogen import AssistantAgent, LLMConfig
from autogen.coding import CodeBlock
from autogen.agentchat.group import AgentNameTarget, StayTarget, ReplyResult
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
executor = JupyterCodeExecutor(server, output_dir=output_dir)

def run_code(code: Annotated[str, "Python code to run in Jupyter"]) -> ReplyResult:
    result = executor.execute_code_blocks(
        [CodeBlock(language="python", code=code)]
    )

    if result.exit_code == 0:
        target = AgentNameTarget("Summariser")
    else:
        target = AgentNameTarget("Summariser")
    msg = f"Exit code: {result.exit_code}\n\nOutput:\n{result.output}\n\nStderr:\n{getattr(result, 'stderr', '')}"
    return ReplyResult(message=msg, target=target)


class Coder(AssistantAgent):
    def __init__(self):

        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4o-mini",
            timeout = 120,
            stream = False,
            # parallel_tool_calls=False
        )
        
        super().__init__(
            name="Coder",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Code Writer. 
                Your role is to take the Planner’s structured step-by-step instructions and write complete, runnable code. 
                Ensure the code is correct, minimal, and follows best practices. 
                Always call the `run_code` tool when you want to write Python code
                If the result indicates there is an error, fix the error and output the code again. 
                Suggest the full code instead of partial code or code changes. 
                If the error can't be fixed or if the task is not solved even after the code is executed successfully, 
                analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
                When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
            """,
            functions=[run_code]
        )
    
    
