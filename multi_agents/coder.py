from pathlib import Path
from typing import Annotated
from autogen import AssistantAgent, LLMConfig
from autogen.coding import CodeBlock
from autogen.agentchat.group import AgentNameTarget, StayTarget, ReplyResult, ContextVariables
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
executor = JupyterCodeExecutor(server, output_dir=output_dir)

def run_code(code: Annotated[str, "Python code to run in Jupyter"], context_variables: ContextVariables) -> ReplyResult:
    result = executor.execute_code_blocks(
        [CodeBlock(language="python", code=code)]
    )

    if result.exit_code == 0:
        target = AgentNameTarget(context_variables["current_agent"])
    else:
        result.output = result.output[:1000]  # truncate long output
        target = AgentNameTarget("Coder")

    msg = f"Exit code: {result.exit_code}\n\nOutput:\n{result.output}\n\nStderr:\n{getattr(result, 'stderr', '')}"
    return ReplyResult(message=msg, target=target)


class Coder(AssistantAgent):
    def __init__(self):

        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-4.1-mini",
            timeout = 120,
            stream = False,
            parallel_tool_calls=False
        )
        
        super().__init__(
            name="Coder",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Coder. 
                Your role is to take the Planner’s structured step-by-step instructions and write complete, runnable code. 
                Ensure the code is correct, minimal, and follows best practices. 
                Always call the `run_code` tool when you want to write Python code
                If the result indicates there is an error, fix the error and output the code again. 
                Use clear print statements so outputs are readable in logs.
                Do NOT create plots, charts, or images.
            """,
            functions=[run_code]
        )
    
    
