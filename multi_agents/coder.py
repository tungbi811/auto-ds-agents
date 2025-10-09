from pathlib import Path
from typing import Annotated
from autogen import AssistantAgent, LLMConfig
from autogen.coding import CodeBlock
from autogen.agentchat.group import AgentNameTarget, ReplyResult, ContextVariables
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
executor = JupyterCodeExecutor(server, output_dir=output_dir, timeout=3600)

def run_code(
        code: Annotated[str, "Python code to run in Jupyter"], 
        context_variables: ContextVariables
    ) -> ReplyResult:
    
    result = executor.execute_code_blocks(
        [CodeBlock(language="python", code=code)]
    )

    if result.exit_code == 0:
        msg = f"Exit code: {result.exit_code}\n\nOutput:\n{result.output}"
        target = AgentNameTarget(context_variables["current_agent"])
    else:
        result.output = result.output[:1000]  # truncate long output
        msg = f"Exit code: {result.exit_code}\n\nOutput:\n{result.output}\n\nStderr:\n{getattr(result, 'stderr', '')}"
        target = AgentNameTarget("Coder")

    return ReplyResult(message=msg, target=target)


class Coder(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type="openai",
            model="gpt-4.1-mini",
            temperature=0.1,
            stream=False,
            parallel_tool_calls=False
        )
        
        super().__init__(
            name="Coder",
            llm_config= llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Coder.
                Your role is to take the structured step instructions and write complete, runnable Python code.

                Rules:
                1. Always ensure the code is minimal, correct, and runnable.
                2. Use clear print() statements so that outputs are easy to understand in logs.
                3. Do not create plots, charts, or images.
                4. When working with pandas:
                - Never use inplace=True. Instead, reassign the result (e.g., df = df.fillna(0) or df['col'] = df['col'].fillna(0)).
                - Prefer .loc for assignments to avoid chained assignment warnings.
                - Use .copy() explicitly when a new DataFrame is intended.
                5. Always call the run_code tool when writing Python code.
                6. If the result indicates an error, fix it and output the corrected code again.

                Rules:
                - Always include the following import at the very beginning of the script to ignore non-critical warnings:
                    import warnings
                    warnings.filterwarnings("ignore")
                - This ensures that RuntimeWarnings or overflow/underflow logs do not clutter the output.
            """,
            functions=[run_code]
        )
    
    
