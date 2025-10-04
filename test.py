from autogen import UserProxyAgent
from multi_agents import BusinessAnalyst, DataExplorer, DataCleaner, FeatureEngineer, Coder, DataEngineer, Modeler, Evaluator, BusinessTranslator
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, RevertToUserTarget
from autogen.agentchat.group.patterns import DefaultPattern
from dotenv import load_dotenv

load_dotenv()

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

ba.handoffs.set_after_work(AgentTarget(data_explorer))
data_explorer.handoffs.set_after_work(AgentTarget(data_cleaner))
data_cleaner.handoffs.set_after_work(AgentTarget(feature_engineer))
feature_engineer.handoffs.set_after_work(AgentTarget(modeler))
modeler.handoffs.set_after_work(AgentTarget(business_translator))
business_translator.handoffs.set_after_work(RevertToUserTarget())

pattern = DefaultPattern(
    initial_agent=ba,
    agents=[ba, data_explorer, data_cleaner, feature_engineer, coder, data_engineer, modeler, evaluator, business_translator],
    user_agent=user,
    group_manager_args = None,
)

# Run the chat
result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="With the dataset uploaded (*.csv), investors want to know which properties deliver the best returns.",
    max_rounds=100
)

# Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.