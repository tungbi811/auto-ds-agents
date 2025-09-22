from autogen import AssistantAgent, LLMConfig
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

### Change this Agent to Data Scientist Agent

# ---------- Training Strategy ----------

class FeatureSelectionStep(BaseModel):
    method: str = Field(..., description="Feature selection method (e.g., SFS, RFE, PCA, variance threshold)")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters of the method (e.g., n_components)")
    rationale: Optional[str] = Field(None, description="Why this method/setting was chosen")

class ClassImbalanceHandling(BaseModel):
    method: Optional[str] = Field(None, description="Technique to handle imbalance (e.g., class_weight, SMOTE)")
    params: Optional[Dict[str, Any]] = Field(None, description="Parameters for the imbalance method")

class HyperparameterTuning(BaseModel):
    method: Optional[str] = Field(
        None, description="Tuning method (e.g., GridSearchCV, RandomizedSearchCV, Bayesian, Optuna)"
    )
    search_space: Optional[Dict[str, Any]] = Field(
        None, description="Key hyperparameters and ranges/distributions"
    )
    n_iter: Optional[int] = Field(None, description="Number of iterations/trials (if applicable)")
    scoring: Optional[str] = Field(None, description="Primary scoring metric")
    cv: Optional[int] = Field(None, description="CV folds used inside tuning method")
    early_stopping: Optional[bool] = Field(False, description="Whether early stopping is used (if supported)")

class CrossValidationPlan(BaseModel):
    strategy: Literal["kfold", "stratified_kfold", "group_kfold", "time_series_split", "holdout"] = Field(
        ..., description="Evaluation split strategy"
    )
    n_splits: Optional[int] = Field(None, description="Number of folds (if CV)")
    shuffle: Optional[bool] = Field(None, description="Shuffle before split (if applicable)")
    random_state: Optional[int] = Field(None, description="Seed for reproducibility")
    group_column: Optional[str] = Field(None, description="Group key for group-based CV")
    holdout_size: Optional[float] = Field(None, description="Holdout ratio for validation/test if not using CV")

class TrainingStrategy(BaseModel):
    description: str = Field(..., description="Detailed description of the training strategy applied")
    cv_plan: CrossValidationPlan = Field(..., description="Cross-validation/holdout plan")
    feature_selection: Optional[List[FeatureSelectionStep]] = Field(None, description="Feature selection steps")
    imbalance_handling: Optional[ClassImbalanceHandling] = Field(None, description="Imbalance technique (if any)")
    tuning: Optional[HyperparameterTuning] = Field(None, description="Hyperparameter tuning configuration")
    data_leakage_guards: Optional[List[str]] = Field(
        None, description="Rules to avoid leakage (fit on train only, time-aware split, etc.)"
    )


# ---------- Best Model ----------

class BestModel(BaseModel):
    model_name: str = Field(..., description="Chosen algorithm/model (e.g., XGBoostClassifier)")
    library: Optional[str] = Field(None, description="Library/framework (e.g., scikit-learn, xgboost)")
    version: Optional[str] = Field(None, description="Library version if relevant")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Key hyperparameters of the best model")
    selected_features: Optional[List[str]] = Field(None, description="Final set of features used by the model")
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Importance scores if available (e.g., feature_importances_, coefficients)"
    )
    training_time_sec: Optional[float] = Field(None, description="Total training duration in seconds")
    rationale: Optional[str] = Field(None, description="Why this model was selected over alternatives")


# ---------- Model Performance (by problem type) ----------

class BinaryConfusionMatrix(BaseModel):
    tn: int = Field(..., description="True negatives")
    fp: int = Field(..., description="False positives")
    fn: int = Field(..., description="False negatives")
    tp: int = Field(..., description="True positives")

class MulticlassConfusionMatrix(BaseModel):
    labels: List[str] = Field(..., description="Class labels order")
    matrix: List[List[int]] = Field(..., description="Confusion matrix counts aligned with labels order")

class PerClassMetrics(BaseModel):
    label: str = Field(..., description="Class label")
    precision: Optional[float] = Field(None, description="Precision for the class")
    recall: Optional[float] = Field(None, description="Recall for the class")
    f1: Optional[float] = Field(None, description="F1 score for the class")
    support: Optional[int] = Field(None, description="Support for the class")

class ClassificationMetrics(BaseModel):
    accuracy: Optional[float] = Field(None, description="Overall accuracy")
    precision_macro: Optional[float] = Field(None, description="Macro-averaged precision")
    recall_macro: Optional[float] = Field(None, description="Macro-averaged recall")
    f1_macro: Optional[float] = Field(None, description="Macro-averaged F1")
    precision_weighted: Optional[float] = Field(None, description="Weighted precision")
    recall_weighted: Optional[float] = Field(None, description="Weighted recall")
    f1_weighted: Optional[float] = Field(None, description="Weighted F1")
    roc_auc: Optional[float] = Field(None, description="ROC-AUC (binary or macro for multi-class)")
    pr_auc: Optional[float] = Field(None, description="PR-AUC (useful for imbalanced problems)")
    specificity: Optional[float] = Field(None, description="Specificity (TNR), if applicable")
    per_class: Optional[List[PerClassMetrics]] = Field(None, description="Per-class metrics")

class RegressionMetrics(BaseModel):
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    mae: Optional[float] = Field(None, description="Mean Absolute Error")
    mse: Optional[float] = Field(None, description="Mean Squared Error")
    r2: Optional[float] = Field(None, description="R-squared")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    medae: Optional[float] = Field(None, description="Median Absolute Error")

class ClusteringMetrics(BaseModel):
    n_clusters: Optional[int] = Field(None, description="Number of clusters used")
    silhouette: Optional[float] = Field(None, description="Silhouette score")
    davies_bouldin: Optional[float] = Field(None, description="Davies–Bouldin index (lower is better)")
    calinski_harabasz: Optional[float] = Field(None, description="Calinski–Harabasz score")
    inertia: Optional[float] = Field(None, description="Within-cluster sum of squares (for k-means)")

class InterpretationNotes(BaseModel):
    narrative: Optional[str] = Field(
        None, description="Detailed interpretation of what the results mean in context"
    )
    key_drivers: Optional[List[str]] = Field(None, description="Most influential features/drivers")
    error_analysis: Optional[List[str]] = Field(None, description="Observed error patterns, failure modes")
    calibration: Optional[str] = Field(None, description="Calibration diagnostics if applicable (e.g., reliability)")

class ModelPerformance(BaseModel):
    problem_type: Literal["classification", "regression", "clustering"] = Field(
        ..., description="Type of ML problem"
    )
    # For classification
    binary_confusion_matrix: Optional[BinaryConfusionMatrix] = Field(
        None, description="Binary confusion matrix if applicable"
    )
    multiclass_confusion_matrix: Optional[MulticlassConfusionMatrix] = Field(
        None, description="Multiclass confusion matrix if applicable"
    )
    classification: Optional[ClassificationMetrics] = Field(
        None, description="Classification metrics"
    )

    # For regression
    regression: Optional[RegressionMetrics] = Field(
        None, description="Regression metrics"
    )

    # For clustering
    clustering: Optional[ClusteringMetrics] = Field(
        None, description="Clustering metrics"
    )

    interpretation: Optional[InterpretationNotes] = Field(
        None, description="Detailed interpretation of metrics and results"
    )


# ---------- Impact of Model Performance ----------

class ImpactItem(BaseModel):
    stakeholder: str = Field(..., description="Stakeholder group (e.g., business, customers, operations)")
    benefits: Optional[List[str]] = Field(None, description="Potential benefits when applying the model")
    drawbacks: Optional[List[str]] = Field(None, description="Potential drawbacks/risks when applying the model")
    operational_changes: Optional[List[str]] = Field(
        None, description="Required process changes to use the model effectively"
    )
    monitoring_kpis: Optional[List[str]] = Field(
        None, description="KPIs to monitor in production (e.g., drift, SLA, business KPIs)"
    )
    risk_mitigation: Optional[str] = Field(None, description="How to mitigate negative impacts")

class ImpactOfModelPerformance(BaseModel):
    description: str = Field(
        ..., description="Overall analysis of how the model’s performance could impact stakeholders"
    )
    items: Optional[List[ImpactItem]] = Field(
        None, description="Stakeholder-wise impact breakdown"
    )


# ---------- Main DS Report ----------

class DataScientistReport(BaseModel):
    training_strategy: TrainingStrategy = Field(
        ..., description="Detailed description of the training strategy applied"
    )
    best_model: BestModel = Field(
        ..., description="Detailed description of the best model and its hyperparameters"
    )
    model_performance: ModelPerformance = Field(
        ..., description="Detailed results and interpretation"
    )
    impact_of_model_performance: ImpactOfModelPerformance = Field(
        ..., description="Business/customer/user impact analysis"
    )


class ModelBuilder(AssistantAgent):
    def __init__(self, llm_config):
        super().__init__(
            name="ModelBuilder",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                # response_format=DataAnalystReport,
            ),
            system_message="""
                You are a Senior Data Scientist. You are responsible for turning the cleaned, processed dataset into a robust 
                and well-evaluated predictive model that directly addresses the business problem defined by the Business Analyst.
            """
        )