from autogen import UserProxyAgent
from multi_agents import BusinessAnalyst, DataExplorer, Coder, DataEngineer
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern

ba = BusinessAnalyst()
user = UserProxyAgent(
    name="User",
    code_execution_config=False
)
data_explorer = DataExplorer()
coder = Coder()
data_engineer = DataEngineer()

pattern = DefaultPattern(
    initial_agent=ba,
    agents=[ba,data_explorer, coder, data_engineer],
    user_agent=user,
    group_manager_args = None,
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="Here is the dataset path: ./data/house_prices/train.csv. Predict for me house price in ./data/house_prices/test.csv",
    max_rounds=100
)