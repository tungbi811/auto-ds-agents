from autogen import AssistantAgent, LLMConfig

class BusinessTranslator(AssistantAgent):
    def __init__(self, llm_config):
        super().__init__(
            name="BusinessTranslator",
            llm_config = LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                temperature=0,
                cache_seed=None
            ),
            system_message="""
                You are the Business Translation Agent.
                Your job is to explain technical results in clear business terms.

                Tasks:
                - Summarize the model results in plain language.
                - Map results back to business objectives and KPIs.
                - Highlight strengths, limitations, risks, and trade-offs.
                - Provide recommendations for deployment or iteration.
                - Suggest monitoring and governance requirements.

                Rules:
                - Write in simple language for non-technical stakeholders.
                - End your FINAL message with <END_OF_WORKFLOW>.
            """
        )
