from multi_agents import BusinessAnalyst, Coder, DataExplorer, DataEngineer, ModelBuilder, Evaluator
from autogen import UserProxyAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern

biz_analyst = BusinessAnalyst()
coder = Coder()
data_explorer = DataExplorer()
data_engineer = DataEngineer()
model_builder = ModelBuilder()
evaluator = Evaluator()

user = UserProxyAgent(
    name="User",
    code_execution_config=False
)

biz_analyst.handoffs.set_after_work(AgentTarget(data_explorer))
data_explorer.handoffs.set_after_work(AgentTarget(coder))

group_chat = DefaultPattern(
    initial_agent=biz_analyst,
    agents=[biz_analyst, coder, data_explorer, data_engineer, model_builder, evaluator],
    user_agent=user
)

initiate_group_chat(
    pattern=group_chat,
    messages="Here is the dataset: data/house_prices/train.csv, Please findout for me if my house have 4 bed and 2 bath, how much is it?",
    max_rounds=20
)