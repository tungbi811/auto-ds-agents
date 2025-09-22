from autogen import AssistantAgent

class Evaluator(AssistantAgent):
    def __init__(self, llm_config):
        super().__init__(
            name="Evaluator",
            llm_config=llm_config,
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
