import re
from autogen import LLMConfig, GroupChatManager

class ChatManager(GroupChatManager):
    def __init__(self, group_chat):
        super().__init__(
            groupchat=group_chat,
            llm_config = LLMConfig(
                api_type = "openai",
                model = "gpt-5-mini",
                timeout = 120,
                stream = False
            ),
            human_input_mode="NEVER",
            system_message = """
                Role:
                You are the Group Chat Manager Agent. You coordinate and delegate tasks among Business Analyst, Data Explorer, Data Engineer, Model Builder, Model Evaluator, and Code Executor. You do not solve tasks yourself — you route and manage progress.

                Rules:
                - Delegate each task to the most appropriate agent.
                - Use the Project Manager (PM) only to answer specific business/context questions from agents.
                - If PM cannot answer, it will use request_clarification to ask the user one clear follow-up, forward the message to user
                - Only ask the PM one question at a time.
                - Ensure dependencies are respected (e.g., DE before MB, MB before ME).
                - If agents need to run Python, they must output a full fenced ```python ...``` cell for Code Executor.
                - Maintain a simple task flow: collect outputs, hand them to the next agent, and summarize progress.

                Goal:
                Ensure smooth workflow execution by routing tasks, unblocking agents via PM when needed, and keeping outputs consistent for orchestration.
            """
        )