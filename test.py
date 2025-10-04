from autogen import UserProxyAgent
from multi_agents import BusinessAnalyst, DataExplorer, DataCleaner, FeatureEngineer, Coder, DataEngineer, Modeler, Evaluator, BusinessTranslator
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group.patterns import DefaultPattern

ba = BusinessAnalyst()
user = UserProxyAgent(
    name="User",
    code_execution_config=False
)
data_explorer = DataExplorer()
data_cleaner = DataCleaner()
feature_engineer = FeatureEngineer()
coder = Coder()
data_engineer = DataEngineer()
modeler = Modeler()
evaluator = Evaluator()
business_translator = BusinessTranslator()

# data_explorer.handoffs.set_after_work(AgentTarget(data_cleaner))

pattern = DefaultPattern(
    initial_agent=ba,
    agents=[ba, coder, data_cleaner],
    user_agent=user,
    group_manager_args = None,
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="""
        Here is the dataset path: ./data/house_prices/train.csv. 
        Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.)
    """,
    max_rounds=100
)