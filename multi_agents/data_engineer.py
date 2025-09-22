from autogen import AssistantAgent, LLMConfig

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ---------- Reusable atoms ----------

class ColumnFix(BaseModel):
    column: str = Field(..., description="Column name this fix applies to")
    method: str = Field(..., description="Method/algorithm used (e.g., IQR capping, Z-score removal, winsorization)")
    params: Optional[Dict[str, Any]] = Field(
        None, description="Key parameters for the method (e.g., iqr_factor, z_threshold)"
    )
    resolved_count: Optional[int] = Field(
        None, description="Count of issues in this column resolved by the method"
    )
    rationale: Optional[str] = Field(None, description="Why this method/setting was chosen for the column")
    notes: Optional[str] = Field(None, description="Additional notes or caveats")


class KeyedDuplicateFix(BaseModel):
    keys: List[str] = Field(..., description="Columns used to define duplicates (composite key allowed)")
    method: str = Field(..., description="Approach used (e.g., drop_duplicates, best-record consolidation)")
    keep: Optional[str] = Field(None, description="Which duplicate to keep (e.g., 'first', 'last')")
    resolved_count: Optional[int] = Field(None, description="Number of duplicate rows removed/merged")
    notes: Optional[str] = Field(None, description="Additional details on tie-breakers or precedence rules")


class MissingFix(BaseModel):
    column: str = Field(..., description="Column with missing values handled")
    strategy: str = Field(..., description="Imputation/removal strategy (e.g., mean, median, mode, MICE, drop)")
    params: Optional[Dict[str, Any]] = Field(None, description="Hyperparameters/settings for the strategy")
    resolved_count: Optional[int] = Field(None, description="Number of missing values resolved")
    insights: Optional[str] = Field(None, description="Notes on missingness pattern (MCAR/MAR/MNAR) and impact")


class VariableIssueFix(BaseModel):
    issue_type: Literal["multicollinearity", "irrelevant_feature", "low_variance",
                        "data_leakage_risk", "encoding_issue", "other"] = Field(
        ..., description="Type of variable-level problem"
    )
    variables: List[str] = Field(..., description="Variables involved in the issue")
    method: str = Field(..., description="Remedy (e.g., VIF filtering, correlation pruning, feature selection)")
    params: Optional[Dict[str, Any]] = Field(None, description="Settings for the method")
    resolved_count: Optional[int] = Field(None, description="Number of variables resolved/removed/adjusted")
    rationale: Optional[str] = Field(None, description="Why this method is appropriate")
    notes: Optional[str] = Field(None, description="Any trade-offs or business considerations")


class EvidenceItem(BaseModel):
    name: str = Field(..., description="Name of evidence (e.g., 'post-clean NA count', 'VIF table')")
    value: Any = Field(..., description="Measured value or summary (can be dict, list, numeric, text)")
    notes: Optional[str] = Field(None, description="Optional commentary")


class SplitParams(BaseModel):
    strategy: Literal["random", "stratified", "time_series", "group"] = Field(..., description="Split strategy")
    test_size: Optional[float] = Field(None, description="Test size ratio (0-1) or absolute count")
    val_size: Optional[float] = Field(None, description="Validation size ratio (if applicable)")
    n_splits: Optional[int] = Field(None, description="Number of folds (for CV)")
    random_state: Optional[int] = Field(None, description="Seed for reproducibility")
    group_column: Optional[str] = Field(None, description="Grouping key for GroupKFold/GroupSplit")
    leakage_guards: Optional[List[str]] = Field(
        None, description="Rules to avoid leakage (e.g., 'split by time', 'split by customer')"
    )


class PipelineStep(BaseModel):
    order: int = Field(..., description="Execution order of this step")
    step_type: Literal["encoding", "scaling_normalization", "transformation", "feature_selection", "other"] = Field(
        ..., description="Type/category of preprocessing step"
    )
    name: str = Field(..., description="Concrete method or transformer name (e.g., OneHotEncoder, StandardScaler)")
    columns: Optional[List[str]] = Field(None, description="Columns targeted (None if applied by column type)")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the method/transformer")
    fit_on: Literal["train_only", "all_data"] = Field(
        "train_only", description="To prevent leakage, most steps should fit on train_only"
    )
    rationale: Optional[str] = Field(None, description="Why this step is needed and chosen")
    artifacts: Optional[List[str]] = Field(None, description="Names/paths of fitted objects to persist")


class TestCase(BaseModel):
    name: str = Field(..., description="Test case name")
    input_brief: str = Field(..., description="What inputs are assumed/provided")
    expected_outcome: str = Field(..., description="What success looks like")
    edge_case: Optional[bool] = Field(False, description="Whether this is an edge case")


class InputSpec(BaseModel):
    sources: List[str] = Field(..., description="Data sources (paths, table names, DataFrame names)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Task parameters (thresholds, seeds, folds)")


class OutputSpec(BaseModel):
    artifacts: List[str] = Field(..., description="Produced outputs (DataFrames, files, fitted objects)")


class CodingTask(BaseModel):
    title: str = Field(..., description="Short imperative title")
    rationale: str = Field(..., description="Why this task is needed")
    priority: Optional[Literal["low", "medium", "high"]] = Field("medium", description="Execution priority")
    dependencies: Optional[List[str]] = Field(None, description="IDs/titles of prerequisite tasks")
    inputs: InputSpec = Field(..., description="Inputs specification")
    outputs: OutputSpec = Field(..., description="Outputs specification")
    acceptance_criteria: List[str] = Field(..., description="Checklist to verify success")
    test_cases: List[TestCase] = Field(..., description="Minimal + edge tests to validate implementation")
    risks_mitigations: Optional[List[str]] = Field(None, description="Key risks and how to mitigate them")


# ---------- Sections tied to your structured output ----------

class OutliersSolution(BaseModel):
    description: str = Field(..., description="Detailed description of methods applied to solve outliers")
    total_resolved: Optional[int] = Field(None, description="Total number of outliers resolved across all columns")
    fixes: Optional[List[ColumnFix]] = Field(None, description="Per-column outlier fixes")


class DuplicatedSolution(BaseModel):
    description: str = Field(..., description="Detailed description of methods applied to solve duplicates")
    total_resolved: Optional[int] = Field(None, description="Total number of duplicate rows resolved")
    keyed_fixes: Optional[List[KeyedDuplicateFix]] = Field(None, description="Key-based duplicate handling details")


class MissingValuesSolution(BaseModel):
    description: str = Field(..., description="Detailed description of methods applied to solve missing values")
    total_resolved: Optional[int] = Field(None, description="Total number of missing values resolved")
    fixes: Optional[List[MissingFix]] = Field(None, description="Per-column missing value handling")


class VariablesSolution(BaseModel):
    description: str = Field(..., description="Detailed description of methods applied to solve variable-level problems")
    total_resolved: Optional[int] = Field(None, description="Number of variables resolved")
    fixes: Optional[List[VariableIssueFix]] = Field(None, description="Detailed fixes for variable-level problems")


class DoubleCheckResult(BaseModel):
    description: str = Field(..., description="Overall description of the post-execution double-check process")
    passed: bool = Field(..., description="True if all checks passed and data is clean")
    evidence: Optional[List[EvidenceItem]] = Field(None, description="Metrics/tables confirming the result")
    remaining_issues: Optional[List[str]] = Field(None, description="Unresolved issues if any")


class SplittingMethod(BaseModel):
    description: str = Field(..., description="Narrative description and rationale for the chosen split")
    params: SplitParams = Field(..., description="Structured split parameters")


class ProcessingPipeline(BaseModel):
    description: str = Field(..., description="Why each preprocessing step is needed")
    steps: List[PipelineStep] = Field(..., description="Ordered preprocessing steps")


class ValidationIteration(BaseModel):
    description: str = Field(..., description="Validation & iteration narrative")
    all_eda_problems_resolved: bool = Field(..., description="Whether all EDA problems are fully resolved")
    unresolved_problems: Optional[List[str]] = Field(None, description="List remaining issues (if any)")
    refined_coding_tasks: Optional[List[CodingTask]] = Field(
        None, description="New/refined tasks for Code Writer to address remaining issues"
    )
    iterations_used: Optional[int] = Field(None, description="How many validation iterations performed")
    ready_for_processing_pipeline: bool = Field(..., description="True if ready to proceed to preprocessing pipeline")


# ---------- Main report ----------

class DataEngineerReport(BaseModel):
    outliers_solution: OutliersSolution
    duplicated_solution: DuplicatedSolution
    missing_values_solution: MissingValuesSolution
    variables_solution: VariablesSolution

    double_check_result: DoubleCheckResult

    splitting_method: SplittingMethod
    processing_pipeline: ProcessingPipeline

    validation_and_iteration: ValidationIteration

    coding_tasks_for_code_writer: List[CodingTask] = Field(
        default_factory=list,
        description="List of coding tasks to be implemented by the Code Writer Agent"
    )


class DataEngineer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="DataEngineer",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
            ),
            system_message="""
                You are a Senior Data Engineer. You design robust pipelines to transform raw datasets into clean, 
                model-ready data. You do not write code directly, but you:
                - Define precise coding tasks for the Code Writer Agent.
                - Ensure the Code Executor successfully applies those tasks.
                - Validate and iterate until all issues identified by the Data Explorer are fully resolved.
                - Only then, define additional preprocessing tasks such as encoding, transformation, feature selection, and normalization.
            """
        )
