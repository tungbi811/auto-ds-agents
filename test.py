from pathlib import Path
from autogen import ConversableAgent, AssistantAgent, LLMConfig, UserProxyAgent, GroupChatManager, GroupChat
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern, DefaultPattern
from autogen.agentchat.group import OnCondition, StringLLMCondition, AgentTarget, RevertToUserTarget, StayTarget
from autogen.coding import LocalCommandLineCodeExecutor

def custom_speaker_selection_func(last_speaker: ConversableAgent, group_chat: GroupChat):
    if len(group_chat.messages) == 1:
        return group_chat.agent_by_name("Data_Explorer")
    
    last_message = group_chat.messages[-1]["content"]
    if "python" in last_message:
        return group_chat.agent_by_name("CodeExecutor")
    elif last_speaker.name == 'CodeExecutor' or last_speaker.name == 'User':
        last_second_speaker_name = group_chat.messages[-2]["name"]
        return group_chat.agent_by_name(last_second_speaker_name)
    return group_chat.agent_by_name("User")

llm_config = LLMConfig.from_json(path="configs/llm_config.json")

output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
executor = JupyterCodeExecutor(server, output_dir=output_dir)

code_executor = ConversableAgent(
    name="CodeExecutor",
    llm_config=False,               # stays tool-only / non-reasoning
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},  # will execute fenced ```python blocks
)

data_explorer = AssistantAgent(
    name="Data_Explorer",
    llm_config=llm_config,
    human_input_mode="NEVER",
    system_message="""You are the data explorer. Given a dataset and a task, please write code to explore and understand the properties of the dataset.
        For example, you can:
        - get the shape of the dataset
        - get the first several rows of the dataset
        - get the information of the dataset use `df.info()` or `df.describe()`
        - plot the plots as needed (i.e. histogram, distribution)
        - check the missing values
        Only perform necessary data exploration steps.
        When you need to execute Python (load CSVs, transform data, plot), write a complete Python cell fenced with ```python ...``` so the CodeExecutor can run it, 
    """,
)

user = UserProxyAgent(name="User",code_execution_config={"use_docker": False})

task_prompt = """Please help me to build a model predict the sales price for each house.
- The train dataset is downloaded to this location: ./data/house_prices/train.csv.
- And then predict dataset in this location: ./data/house_prices/test.csv
"""

pattern = GroupChat(
    agents=[user, data_explorer,code_executor],
    speaker_selection_method=custom_speaker_selection_func,
    max_round=20
)

group_chat_manager = GroupChatManager(pattern, llm_config=llm_config)


group_chat_manager.handoffs.add_llm_conditions([
    OnCondition(
        target=AgentTarget(code_executor),
        condition=StringLLMCondition(prompt="When response have code")
    )
])

result = user.initiate_chat(group_chat_manager, message=task_prompt)

# Run the chat
# result, final_context, last_agent = initiate_group_chat(
#     pattern=pattern,
#     messages=task_prompt,
#     max_rounds=10,
# )