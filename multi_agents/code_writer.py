from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class ExecutionRequest(BaseModel):
    description: str = Field(
        ..., 
        description="Please execute the following code for me"
    )
    requester: Optional[str] = Field(
        None, description="Agent or component requesting execution (e.g., Code Writer Agent)"
    )
    run_context: Optional[str] = Field(
        None, description="High-level context: 'cleaning', 'training', 'evaluation', etc."
    )


class CodeMetadata(BaseModel):
    task: str = Field(
        ..., 
        description='Brief task requirement, e.g., "Remove duplicate rows from dataset"'
    )
    description: str = Field(
        ..., 
        description="Step-by-step explanation of the logic: what is done, why it is needed, expected outcome"
    )
    additional_notes: Optional[str] = Field(
        None, 
        description="Parameters, assumptions, or libraries used"
    )
    libraries: Optional[List[str]] = Field(
        None, description="Explicit list of required libraries/imports"
    )
    inputs: Optional[Dict[str, Any]] = Field(
        None, description="Optional I/O specification: sources, schema, parameters"
    )
    outputs: Optional[Dict[str, Any]] = Field(
        None, description="Optional I/O specification: artifacts, result schema"
    )
    validation_notes: Optional[str] = Field(
        None, description="How this code follows validation rules (schema checks, reproducibility, leakage guards)"
    )
    version_tag: Optional[str] = Field(
        None, description="Optional code version tag or hash for traceability"
    )


class CodeBlock(BaseModel):
    language: Literal["python"] = Field(
        "python", description="Programming language of the fenced code block"
    )
    body: str = Field(
        ..., 
        description="Raw code content (to be enclosed by the executor in a ```python ... ``` fenced block)"
    )


class CodeWriterOutput(BaseModel):
    """
    Structured output for the Code Writer Agent.
    """
    execution_request: ExecutionRequest = Field(
        ..., description="Execution request header"
    )
    code_metadata: CodeMetadata = Field(
        ..., description="Human-readable explanation and metadata of the code"
    )
    code: CodeBlock = Field(
        ..., description="Executable code block"
    )

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
    
    
