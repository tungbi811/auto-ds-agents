import re
from pathlib import Path
from typing import Annotated
from autogen import AssistantAgent, LLMConfig
from autogen.coding import CodeBlock
from autogen.agentchat.group import AgentNameTarget, ReplyResult, ContextVariables, RevertToUserTarget
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

server = LocalJupyterServer(
    log_file='./logs/jupyter_gateway.log',
)
executor = JupyterCodeExecutor(server, output_dir=output_dir, timeout=1200)

def run_code(
        code: Annotated[str, "Python code to run in Jupyter"], 
        context_variables: ContextVariables
    ) -> ReplyResult:
    try:
        result = executor.execute_code_blocks(
            [CodeBlock(language="python", code=code)]
        )
    except Exception as e:
        executor.restart()
        result = executor.execute_code_blocks(
            [CodeBlock(language="python", code=code)]
        )
    except Exception as e:
        return ReplyResult(message=f"Execution failed: {e}", target=RevertToUserTarget())

    if result.exit_code == 0:
        msg = f"Output:\n{result.output}"
        target = AgentNameTarget(context_variables["current_agent"])
    else:
        # Split into lines and clean
        lines = result.output.splitlines()

        errors = []
        for line in lines:
            errors.append(line)
            if len(errors) >= 5:
                break
        msg = "\n".join(errors)
    return ReplyResult(message=msg, target=target)


class Coder(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type="openai",
            model="gpt-4.1-mini",
            temperature=0,
            stream=False,
            parallel_tool_calls=False
        )
        
        super().__init__(
            name="Coder",
            llm_config= llm_config,
            human_input_mode="NEVER",
            system_message = """
                You are the Coder agent.

                Your mission is to take structured analytical step instructions and implement them as complete, runnable Python code within a Jupyter Notebook environment.

                Environment:
                - You are working in a Jupyter Notebook, where common libraries (pandas, numpy, sklearn, etc.) may already be imported.
                - Do not re-import standard libraries unless the required module has not been imported yet.
                - Avoid redefining or recreating variables that already exist unless explicitly instructed to do so.
                - Reuse context variables passed to you when appropriate.

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
                7. Never leave a variable name or expression alone on the last line of the code cell.
                - Always print() the values you want to display.
                - If you need to show multiple outputs, print summaries or formatted outputs.
                - Never rely on implicit return display (e.g., do not end with 'df' or 'model').

                Standard Imports:
                - Always include the following import at the top of your script to suppress non-critical warnings:
                    import warnings
                    warnings.filterwarnings("ignore")
                - This ensures RuntimeWarnings and overflow/underflow logs do not clutter the output.

                Behavioral Guidelines:
                - Write clear, readable, and reproducible code.
                - Comment complex logic briefly and meaningfully.
                - Maintain consistent variable names throughout the workflow.
                - Focus on correctness and clarity over optimization or brevity.
            """,
            functions=[run_code]
        )
    
    
