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
