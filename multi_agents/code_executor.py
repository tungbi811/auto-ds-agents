from pathlib import Path
from autogen import ConversableAgent
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

from typing import Optional, Any
from pydantic import BaseModel, Field


class CodeExecutionRequest(BaseModel):
    code: str = Field(
        ..., 
        description="The code script that you are asked to execute"
    )


class CodeExecutionResult(BaseModel):
    success_execution: Optional[Any] = Field(
        None, 
        description="Full execution results of the code script (printed outputs, returned objects, logs, plots, etc.)"
    )
    error: Optional[str] = Field(
        None, 
        description="Full error message/stack trace if execution failed (None if successful)"
    )


class CodeExecutorOutput(BaseModel):
    """
    Structured output for the Code Executor Agent.
    """
    code: CodeExecutionRequest = Field(
        ..., description="Code block to be executed"
    )
    result: CodeExecutionResult = Field(
        ..., description="Execution outcome: either success results or error message"
    )


class CodeExecutor(ConversableAgent):
    def __init__(self):
        output_dir = Path("./artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
        executor = JupyterCodeExecutor(server, output_dir=output_dir)

        super().__init__(
            name="CodeExecutor",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config={"executor": executor},
        )
