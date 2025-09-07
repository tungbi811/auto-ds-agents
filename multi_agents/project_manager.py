from autogen import ConversableAgent

class ProjectManager(ConversableAgent):
    def __init__(self, llm_config):
        super().__init__(
            name="ProjectManager",
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message=""
        )