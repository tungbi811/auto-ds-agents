from autogen import UserProxyAgent
from multi_agents import BusinessAnalyst, DataExplorer, Coder, DataEngineer, Modeler
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
modeller = Modeler()
pattern = DefaultPattern(
    initial_agent=ba,
    agents=[ba, data_explorer, coder, data_engineer, modeller],
    user_agent=user,
    group_manager_args = None,
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="Here is the dataset path: ./data/house_prices/train.csv. Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.)",
    max_rounds=100
)