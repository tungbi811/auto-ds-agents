from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import ContextVariables, ReplyResult, AgentNameTarget
from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class DataExploringPlan(BaseModel):
    plan_description: str = Field(
        ...,
        description=(
            "A concise description of the overall exploration plan. It should outline the key steps to take in order to understand the dataset."
        ),
        example=(
            "Follow a business-aligned EDA plan: (1) audit null values by column and row, quantify rates, "
            "and propose per-column handling; (2) detect duplicated rows/keys and propose de-duplication; "
            "(3) validate domain/type consistency (e.g., 'height' not as '9-May', fixable parsing rules; "
            "'date' like '20251029' → '2025/10/29'); (4) identify numeric outliers; if this is a "
            "classification task, separate features that include the target from those that do not to guide "
            "outlier removal decisions; (5) analyze relationships with the target using methods appropriate "
            "to task type: for classification use Chi-square + Cramér’s V for categorical vs target and "
            "ANOVA/Kruskal for numeric vs target; for regression use correlation matrix (Pearson/Spearman) "
            "and partial diagnostics. Summarize issues and propose high-level handling options."
        )
    )

    columns: Optional[List[str]] = Field(
        None,
        description=(
            "List of specific columns to apply the step to. "
            "If None, apply the step to all relevant columns automatically."
        ),
        example=["age", "income", "gender"]
    )

class Insight(BaseModel):
    column: str = Field(
        ...,
        description="The name of the column where the insight was found.",
        example="age"
    )
    insight: str = Field(
        ...,
        description="A concise description of the insight discovered.",
        example="The age distribution is right-skewed with a median of 35."
    )

class Issue(BaseModel):
    column: str = Field(
        ...,
        description="The name of the column where the issue was found.",
        example="income"
    )
    issue: str = Field(
        ...,
        description="A concise description of the data quality issue discovered.",
        example="There are 15% missing values in this column."
    )
    recommendation: str = Field(
        ...,
        description="A recommended action to address the identified issue.",
        example=(
            "Consider imputing missing values with the median or removing rows with missing data.",
            "Remove outliers using the IQR method."
            ""
        )
    )


class DataExplorerOutput(BaseModel):
    insights: List[Insight] = Field(
        ...,
        description="A list of key insights discovered during data exploration."
    )
    issues: List[Issue] = Field(
        ...,
        description="A list of data quality issues identified during data exploration."
    )
    recommended_next_steps: List[str] = Field(
        ...,
        description="A list of recommended next steps for data cleaning or feature engineering based on the findings."
    )

def execute_data_exploring_plan(
    plan: DataExploringPlan,
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Delegate coding of a specific exploration step to the Coder agent.
    """
    context_variables["current_agent"] = "DataExplorer"
    return ReplyResult(
        message=(
                f"Please write Python to execute this exploration step and return structured findings.\n\n"
                f"Exploration step (as plan object): {plan}\n\n"
                "Requirements for your response:\n"
                "1) Print concise tables/plots as needed (missingness, duplicates, tests).\n"
                "2) Return three lists we can map into our schema:\n"
                "   - insights: List[Insight(column, insight)] with clear, column-scoped observations.\n"
                "   - issues: List[Issue(column, issue, recommendation)] where recommendation states a high-level handling option.\n"
                "   - recommended_next_steps: List[str] (prioritized, actionable EDA-driven suggestions for the DataCleaner).\n"
                "3) Methods to apply depending on the step:\n"
                "   - Nulls: per-column/row rates, co-missingness; propose impute/drop rules.\n"
                "   - Duplicates: exact row dupes and key dupes; propose dedup strategy.\n"
                "   - Domain/Type: detect invalid parses (e.g., 'height' as '9-May', 'date' as '20251029'); "
                "     propose parsing rules (e.g., convert to 5.09 feet; parse to YYYY/MM/DD) and valid ranges.\n"
                "   - Outliers: IQR and robust z-scores; if classification, separate features tied to target vs others for removal decisions.\n"
                "   - Relationships:\n"
                "       * Classification: Chi-square + Cramér’s V for cat–target; ANOVA/Kruskal for num–target.\n"
                "       * Regression: Pearson/Spearman correlation matrix for predictors vs numeric target.\n"
                "4) Include effect sizes and p-values where relevant, and note multiple-testing caveats.\n"
            ),
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_data_explore_task(
    results: DataExplorerOutput,
    context_variables: ContextVariables
) -> ReplyResult:
    """
    Complete the DataExplorer stage and hand off results to the DataEngineer.
    """
    context_variables["data_insights"] = results.insights
    context_variables["data_issues"] = results.issues
    context_variables["current_agent"] = "DataCleaner"
    return ReplyResult(
        message=f"Here are the recommended next steps: {results.recommended_next_steps}",
        target=AgentNameTarget("DataCleaner"),
    )

class DataExplorer(AssistantAgent):
    def __init__(self):
        super().__init__(
            name = "DataExplorer",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4.1-mini",
                response_format=DataExplorerOutput,
                parallel_tool_calls=False
            ),
            system_message = """
                    You are the DataExplorer.

                    Goal:
                    - Inspect the dataset strictly following this business EDA checklist, then summarize findings.

                    Checklist (run all, step-by-step):
                    1) Null Values
                    - Compute missingness per column and per row; flag columns above 0%, 5%, 10%.
                    - Identify patterns of co-missingness (columns that often go missing together).
                    - For each affected column, propose handling options (e.g., drop, median/most-frequent impute, model-based impute), and note risks.

                    2) Duplicates
                    - Report exact duplicate rows and duplicate keys (if candidate keys exist).
                    - Provide counts and example records.
                    - Propose deduplication approach (keep-first, keep-latest by timestamp, aggregate rules).

                    3) Domain / Type Validation
                    - Detect values inconsistent with column meaning (e.g., 'height' parsed as date '9-May'; 'date' stored as '20251029').
                    - Suggest canonical parsing/conversion rules (e.g., height '5.09 ft' normalization; date 'YYYY/MM/DD').
                    - List columns with mixed types, invalid ranges, or out-of-domain categories.

                    4) Outliers (Numeric)
                    - Detect per-feature outliers using robust methods (IQR, MAD-based z-scores).
                    - If task type is classification:
                        • Split results into: (a) features that are the target (or derived from it) vs (b) non-target features.
                        • Provide guidance: when removal is safe vs risky for target-bearing signals.
                    - If task type is regression:
                        • Quantify influence/extremes and potential impact on linear assumptions.

                    5) Relationships with the Target (choose methods by task type)
                    - Classification:
                        • Categorical vs target: Chi-square test + Cramér’s V (report effect sizes and p-values).
                        • Numeric vs target: ANOVA (or Kruskal if non-normal), plus simple group means/variance.
                    - Regression:
                        • Correlation matrix (Pearson and Spearman) for numeric predictors vs numeric target.
                        • Flag multicollinearity clusters relevant to target interpretation (brief).
                    - Always mention key caveats (multiple testing, non-linearity).

                    Output Contract:
                    - Do NOT clean or engineer features. Only explore and summarize.
                    - For each step, delegate coding to the Coder via `execute_data_exploring_plan`.
                    - Aggregate results into `DataExplorerOutput`:
                        • insights: concise, column-scoped insights.
                        • issues: each with column, issue, recommendation (high-level handling path).
                        • recommended_next_steps: prioritized EDA-driven actions for the DataCleaner.
                    - When done, call `complete_data_explore_task`.

                    Workflow:
                    1) Build a structured DataExploringPlan from BusinessAnalyst context.
                    2) For each plan step, call `execute_data_exploring_plan` (no self-coding).
                    3) Collect results and produce `DataExplorerOutput`.
                    4) Call `complete_data_explore_task`.

                    Rules:
                    - You must call two functions provided: `execute_data_exploring_plan` and `complete_data_explore_task`.
                    - Don't perform coding yourself. Always delegate coding to the Coder agent.
                    - Don't make a plan for actual cleaning, feature engineering, or modeling. Focus only on exploration and summarization.
            """,
            functions=[execute_data_exploring_plan, complete_data_explore_task]
        )
