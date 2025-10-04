from multi_agents import BusinessAnalyst, Coder, DataExplorer, DataCleaner, Modeler, Evaluator, FeatureEngineer
from autogen import UserProxyAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables
from autogen.agentchat.group.patterns import DefaultPattern

context_variables = ContextVariables(data={
    "current_agent": "",
    "objective": "",
    "problem_type": "",
    "research_questions": [],
    "data_issues": [],
    "data_insights": [],
})

biz_analyst = BusinessAnalyst()
coder = Coder()
data_explorer = DataExplorer()
data_cleaner = DataCleaner()
feature_engineer = FeatureEngineer()
modeler = Modeler()

user = UserProxyAgent(
    name="User",
    code_execution_config=False
)

group_chat = DefaultPattern(
    initial_agent=biz_analyst,
    agents=[biz_analyst, coder, data_explorer, data_cleaner, feature_engineer, modeler],
    user_agent=user,
    context_variables=context_variables
)

initiate_group_chat(
    pattern=group_chat,
    messages="""
        Here is the dataset path: ./data/house_prices/train.csv. 
        Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.)
    """,
    max_rounds=100
)