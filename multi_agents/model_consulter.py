from autogen import AssistantAgent, LLMConfig
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class ModelBuilder(AssistantAgent):
    def __init__(self):
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