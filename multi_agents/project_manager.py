from autogen import LLMConfig, AssistantAgent
from utils.utils import request_clarification

class ProjectManager(AssistantAgent):
    def __init__(self):
        llm_config = LLMConfig(
            api_type = "openai",
            model = "gpt-5-mini",
            timeout = 120,
            stream = False
        )
        super().__init__(
            name="ProjectManager",
            llm_config = llm_config,
            human_input_mode="NEVER",
            system_message = """
                Role:
                You are the Project Manager Agent. Your job is to answer agents’ business or context questions. You do not manage tasks or delegate work.

                Rules:
                - Provide clear, concise answers to agent questions based on business objectives.
                - If you cannot answer, use the request_clarification tool to ask the user one targeted follow-up question.
                - Only ask one clarification at a time, then wait for the answer before continuing.
                - Do not perform technical tasks or write code — focus only on business and context clarification.

                Goal:
                Act as the oracle of business requirements and context so other agents can continue their work without confusion.
            """,
            functions = [request_clarification]
        )