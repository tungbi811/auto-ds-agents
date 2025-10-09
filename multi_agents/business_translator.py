from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, RevertToUserTarget,ReplyResult
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


class BusinessTranslatorStep(BaseModel):
    step_description: str = Field(
        ...,
        description= "Type of business translation step to perform.",
        examples=[
                "Reload the best model from the specified path ./artifacts/best_model_{problem_type}.joblib and use it to generate predictions for user-provided data point."
                "If the user data point includes EXTRA features not used in training, DROP the extra columns and proceed; record them as 'ignored_features'.",
                "If the user data point is MISSING required features, fill them using the same imputation strategy as training (or set to NaN and let the pipeline handle it); record them as 'missing_features'.",
                "=== Classification ===\n"
                "- Run the best classification model on the user’s data point; if features are missing, enrich with related variables to improve prediction; If the user data has EXTRA columns not present in training, DROP them.\n"
                "- Interpret model output (e.g., predicted class, probability) and explain top features driving the decision.\n"
                "- Translate the prediction into business meaning — e.g., likelihood of churn and factors influencing it.\n\n"
                "=== Regression ===\n"
                "- Check if user data point is incomplete, estimate missing variables from known correlations or imputation; If the user data has EXTRA columns not present in training, DROP them.\n"
                "- Predict target value (e.g., sales, rainfall, price) using the best regression model base one.\n"
                "- Analyze sensitivity of the prediction to top influencing variables and explain their economic meaning.\n"
                "- Summarize key drivers, potential scenarios, and expected business outcomes.\n\n"
                "=== Clustering ===\n"
                "- Analyze cluster profiles based on the initial dataset and centroid statistics.\n"
                "- Determine which cluster the user's data point belongs to, based on nearest centroid distance.\n"
                "- Describe defining characteristics of that cluster (e.g., average income, age range, behavior).\n"
                "- Summarize opportunities and risks for business actions related to this cluster.\n\n"
                "=== Time Series ===\n"
                "- Generate forecasts for the next H periods using the trained time series model.\n"
                "- Decompose the forecast into trend, seasonality, and residual components.\n"
                "- Identify anomalies or sudden structural changes in the series.\n"
                "- If exogenous variables (X) are used, explain their contribution to the forecast.\n"
                "- Provide scenario-based simulations (e.g., +10% marketing spend, -5% demand) and recommend business actions such as inventory, staffing, or budgeting adjustments."
        ]
    )
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
                "Reload the best model from the specified path (e.g., ./artifacts/best_model_{problem_type}.pkl) and use it to generate predictions for a new user-provided data point."
                "=== Classification ===\n"
                "1. Use the trained classification model to predict the class and probability for the user’s provided data point.\n"
                "2. If important features are missing, enrich the input with correlated variables to improve accuracy; If the user data has EXTRA columns not present in training, DROP them.\n"
                "3. Analyze the top features influencing the prediction (e.g., SHAP values or feature importance).\n"
                "4. Translate these drivers into business meaning, explaining why a certain outcome (e.g., churn) is likely.\n"
                "5. Provide concrete business recommendations to improve the likelihood of a positive outcome.\n\n"
                "=== Regression ===\n"
                "1. Run the trained regression model to estimate the target value for the user’s data point.\n"
                "2. Handle missing variables using interpolation or appropriate imputation techniques; If the user data has EXTRA columns not present in training, DROP them.\n"
                "3. Evaluate how each independent variable impacts the predicted value and interpret it economically.\n"
                "4. Discuss the potential business implications, benefits, and risks of the prediction results.\n\n"
                "=== Clustering ===\n"
                "1. Do not rerun the clustering model — use the existing segmentation output.\n"
                "2. Assign the user’s data point to the nearest cluster by computing the distance to the centroid.\n"
                "3. Calculate key statistics (mean, median) of variables within the assigned cluster.\n"
                "4. Describe the main characteristics that define this cluster and how it differs from others.\n"
                "5. Evaluate opportunities and business risks associated with this segment and propose relevant actions.\n\n"
                "=== Time Series ===\n"
                "1. Apply the trained forecasting model to predict values for the next H time steps.\n"
                "2. Decompose the forecast into trend, seasonality, and residual components to reveal underlying patterns.\n"
                "3. Identify anomalies or regime shifts that could signal changes in business performance.\n"
                "4. Interpret the influence of exogenous variables (if applicable) and conduct scenario simulations (e.g., +10% marketing budget). "
                "Provide actionable recommendations such as adjusting inventory, workforce planning, or investment timing."
        ]
    )

    reason: str = Field(
        ...,
        description="Purpose and rationale for performing this step.",
        examples=[
                "Ensure that analytical insights are directly tied to the business question, clearly explained, and "
                "supported by evidence. This step helps the user understand the reasoning behind conclusions and "
                "recognize the potential risks and strategic implications of recommended business actions."
        ]
    )


def execute_business_translation_step(
    step: BusinessTranslatorStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate coding of a specific business translator step to the Coder agent.
    """
    context_variables["current_agent"] = "BusinessTranslator"
    return ReplyResult(
        message=f"Please write Python code to execute this business translator step:\n{step.step_description} - {step.instruction}",
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )

def complete_business_translation_task(
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message=f"Business translation is complete.",
        target=RevertToUserTarget(),
        context_variables=context_variables,
    )

class BusinessTranslator(ConversableAgent):
    def __init__(self):
        super().__init__(
            name="BusinessTranslator",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-5-mini",
                # parallel_tool_calls=False,
                # temperature=0.3,
            ),
            update_agent_state_before_reply= [UpdateSystemMessage(
            """
            You are the BusinessTranslator.
            Your role is to interpret analytical results and translate them into clear, actionable business insights and strategic recommendations tailored to stakeholder needs.
            You ensure that analytical findings are directly connected to business objectives, stakeholder expectations, and measurable outcomes.

            Stakeholder Expectations:
            {stakeholders_expectations}

            Key Responsibilities:
            - Understand the business context and intent behind the research question.
            - Review analytical outputs provided by other agents to extract key insights that directly address the stakeholder’s question.
            - Communicate findings in plain business language, focusing on their practical meaning and implications.
            - **Clarity and focus** — respond directly to the stakeholder’s intent; avoid technical language or lengthy explanations.
            - **Insight-driven** — highlight only the most essential patterns or segment characteristics that explain the outcome.
            - **Action-oriented** — for each key insight, propose what the business should *do* (e.g., target, invest, optimize, mitigate) and outline potential benefits or risks.
            - **Generalizable** — ensure the output format applies to any business question or dataset (e.g., segmentation, forecasting, classification).
            - **Plain business language** — never mention algorithms, preprocessing steps, or statistical terminology.

            Workflow:
            1) Understand the user’s intent and whether a specific data point was provided (and which features are present/missing).
            2) Review available analytical outputs (models, centroids, profiles) from other agents relevant to the question.
            3) Branch:
            - For classification/regression/time_series with user data:
                • Build a BusinessTranslatorStep that tells the Coder to reload ./artifacts/best_model_{problem_type}.joblib and predict for that data point (no retraining).
                • Call execute_business_translation_step to send that step to the Coder.
            - For clustering:
                • Do NOT call any predictive model. Determine the best-fit cluster by comparing the user’s features with existing cluster profiles/centroids.
                • (Optional) If numeric distance is necessary, you may call the Coder for a simple nearest-centroid computation—no model loading.
            4) Convert the technical outcome into business terms: key insight(s), what it means, and what the business should do next.
            5) Summarize the final output concisely under headings like **Key Insights** and **Business Recommendations**.

            Rules:
            - Do not include technical details such as algorithms, data preprocessing, or modeling methods.
            - Do not use any technical terms such as “regression,” “cluster,” “p-value,” “confidence interval,” or “feature importance.”
            - A strong understanding of the underlying characteristics provides the foundation for meaningful business insights.
            - Focus on clarity, relevance, and real-world applicability.
            - Ensure that each recommendation or action plan directly aligns with both the research question and stakeholder expectations.
            - Use persuasive, professional, and business-oriented language suitable for decision-makers.
            - Maintain a concise and results-driven tone.
            - Ensure every insight and recommendation connects logically to the question and stakeholder goals.
            """

            )
            ],           
            functions=[execute_business_translation_step]
        )

