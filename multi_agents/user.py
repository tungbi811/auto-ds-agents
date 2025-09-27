from autogen import UserProxyAgent, LLMConfig

def is_termination_msg_chat(message:str) -> bool:
    """
    Checks if the message from the last agent was a termination message or not.

    Args:
        message (str): last message from the agent
    Return:
        state (boolean): true if the message is empty or if ends with TERMINATE, False otherwise
    """

    if isinstance(message, dict):
        message = message.get("content", "")

    termination_keywords = "TERMINATE"
    state = message.rstrip().endswith(termination_keywords) or not message.rstrip()
    return state

class User(UserProxyAgent):
    def __init__(self):
        super().__init__(
            name="User",
            llm_config=LLMConfig(
                api_type= "openai",
                model="gpt-4o-mini",
                temperature=0,
                cache_seed=None
            ),
            code_execution_config={
                "last_n_messages": 2,
                "work_dir": "tasks",
                "use_docker": False,
            },
            human_input_mode="NEVER",
            system_message="""
                You are the user of a data science project.
                You will provide the necessary information and feedback to the agents involved in the project.
                You will interact with the agents to ensure that your requirements are met and that the project progresses smoothly.
                You will provide feedback on the agents' outputs and guide them towards achieving your goals.
                You will respond to the agents' requests for information and clarification.
                You will make decisions based on the information provided by the agents.
                You will ensure that the project stays on track and meets your expectations.
            """,
            is_termination_msg=is_termination_msg_chat,
        )