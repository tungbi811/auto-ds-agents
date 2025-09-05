from autogen import ConversableAgent

class DataEngineer(ConversableAgent):
    def __init__(self, llm_config):
        super().__init__(
            name="DataEngineer",
            llm_config=llm_config,
            system_message="""
                You are the Data Engineer Agent (CRISP-DM: Data Preparation).
                Your job is to prepare clean, usable data for modeling.

                Tasks:
                - Clean, transform, and integrate data sources.
                - Engineer features and create training/validation/test splits.
                - Validate outputs (schema, ranges, leakage checks).
                - Save prepared datasets and artifacts to ./artifacts.
                - Provide handoff notes for the Model Builder Agent.

                Rules:
                - Do not overwrite raw data; always produce derived outputs.
                - If you include code, provide it in a fenced Python block and end with <RUN_THIS>.
            """
        )
