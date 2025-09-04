from autogen import GroupChatManager

class Manager(GroupChatManager):
    def __init__(self, group_chat, llm_config):
        super().__init__(
            groupchat=group_chat,
            llm_config=llm_config,
            system_message="""
                You are the manager of this multi-agent workflow. 
                Your job is to decide which agent should speak next, following these rules:

                1. **Initialization**
                - If the last speaker was the initializer, always send the turn to **Data_Explorer**.

                2. **Exploration / Processing / Training (two-step states)**
                - If the last speaker was **Data_Explorer**, **Data_Processer**, or **Model_Trainer**, 
                    always route next to **Code_Executor**.

                3. **Code Execution results**
                - If the last speaker was **Code_Executor**:
                    - If the last message contains `"exitcode: 1"`, then repeat the step by routing 
                    back to the agent who spoke before the Code_Executor.
                    - If the agent before Code_Executor was **Data_Explorer**, route next to **Data_Processer**.
                    - If the agent before Code_Executor was **Data_Processer**:
                    - If the data is ready for training, route to **Model_Trainer**.
                    - Otherwise, go back to **Data_Explorer**.
                    - If the agent before Code_Executor was **Model_Trainer**:
                    - If the number of training trials is fewer than 2, route again to **Model_Trainer**.
                    - Otherwise, route to **Code_Summarizer**.

                4. **Summarization**
                - If the last speaker was **Code_Summarizer**, the workflow ends. Do not select another speaker.

                Important:
                - Never choose yourself as the next speaker.
                - Follow the above rules exactly; do not improvise new transitions.
                """
        )