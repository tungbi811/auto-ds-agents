from multi_agents import BusinessAnalyst, Coder, DataExplorer, DataEngineer, ModelBuilder, Evaluator
from autogen import UserProxyAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables
from autogen.agentchat.group.patterns import DefaultPattern

context_variables = ContextVariables(
    data={
        "user_question": None,
        ""
        "current_agent": None
    }
)

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
data_explorer.handoffs.set_after_work(AgentTarget(data_engineer))

group_chat = DefaultPattern(
    initial_agent=data_explorer,
    agents=[biz_analyst, coder, data_explorer, data_engineer, model_builder, evaluator],
    user_agent=user,
    context_variables=context_variables
)

initiate_group_chat(
    pattern=group_chat,
    messages="Here is the dataset: data/house_prices/train.csv, build for me a predictive model for my house have 4 bed and 2 bath, how much is it?",
    max_rounds=100
)