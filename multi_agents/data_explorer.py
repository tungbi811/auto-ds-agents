from autogen import AssistantAgent, LLMConfig
from pydantic import BaseModel, Field
from typing import List, Optional

class DataQualityIssue(BaseModel):
    column: str = Field(..., description="Column name where the issue occurs")
    issue_type: str = Field(..., description="Type of issue (missing, outlier, skew, leakage risk, etc.)")
    severity: str = Field(..., description="Severity rating: low, medium, high")
    details: Optional[str] = Field(None, description="Additional explanation of the issue")

class ColumnSummary(BaseModel):
    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type (int, float, string, timestamp, etc.)")
    missing_pct: float = Field(..., description="Percentage of missing values (0-100)")
    unique_values: Optional[int] = Field(None, description="Number of unique values, if categorical")
    example_values: Optional[List[str]] = Field(None, description="Sample values for quick inspection")

class DataUnderstandingOutput(BaseModel):
    dataset_name: str = Field(..., description="Name or ID of the dataset being profiled")
    num_rows: int = Field(..., description="Total number of rows")
    num_columns: int = Field(..., description="Total number of columns")
    
    columns: List[ColumnSummary] = Field(
        ..., description="Summary information for each column"
    )
    quality_issues: List[DataQualityIssue] = Field(
        default_factory=list,
        description="List of data quality issues detected"
    )
    
    anomalies: Optional[List[str]] = Field(
        None, description="Notable anomalies, such as extreme values or unusual patterns"
    )
    potential_leakage: Optional[List[str]] = Field(
        None, description="Columns that might leak target information"
    )
    
    overall_notes: Optional[str] = Field(
        None, description="General notes or impressions from data exploration"
    )
    next_steps: Optional[List[str]] = Field(
        None, description="Recommended actions to deepen data understanding"
    )

class DataExplorer(AssistantAgent):
    def __init__(self, llm_config):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
            ),
            system_message = """
                Explore dataset by using python 
                Explain the rationale on the definition of the target variable according to your business use case.
            """
        )

