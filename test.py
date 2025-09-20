from autogen import UserProxyAgent
from multi_agents import BusinessAnalyst, DataExplorer
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern

ba = BusinessAnalyst()
data_explorer = DataExplorer()
user = UserProxyAgent(
    name="User",
    code_execution_config=False
)
ba.handoffs.set_after_work(AgentTarget(data_explorer))

pattern = DefaultPattern(
    initial_agent=ba,
    agents=[
        ba,
        data_explorer,
    ],
    user_agent=user,
    group_manager_args = None,
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="I want to sell my house but i don't know the price",
    max_rounds=15
)