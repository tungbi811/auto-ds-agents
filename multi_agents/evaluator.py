from autogen import AssistantAgent, LLMConfig

class Evaluator(AssistantAgent):
    def __init__(self):
        super().__init__(
            name="Evaluator",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
            ),
            system_message="""
                You are the Evaluator Agent
                Your job is to validate whether the chosen model meets acceptance criteria.

                Tasks:
                - Recompute metrics on the test set.
                - Run error analysis and segment analysis.
                - Check robustness, fairness, and operational constraints.
                - Decide if the model is acceptable or if iteration is needed.
                - Provide handoff notes for the Business Translation Agent.

                Rules:
                - Tie evaluation back to business objectives.
                - If you include code, provide it in a fenced Python block and end with <RUN_THIS>.
            """
        )
