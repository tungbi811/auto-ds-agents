from autogen import UserProxyAgent
from multi_agents import BusinessAnalyst, DataExplorer
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern

ba = BusinessAnalyst()
user = UserProxyAgent(
    name="User",
    code_execution_config=False
)

pattern = DefaultPattern(
    initial_agent=ba,
    agents=[ba],
    user_agent=user,
    group_manager_args = None,
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="Here is the dataset path: ./data/house_prices/train.csv . Predict for me the average salary",
    max_rounds=15
)