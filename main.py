from multi_agents import BusinessAnalyst, DataAnalyst, DataEngineer, Coder
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

coder = Coder()
business_analyst = BusinessAnalyst()
data_anayst = DataAnalyst()
data_engineer = DataEngineer()

user = UserProxyAgent(
    name="User",
    code_execution_config=False
)

group_chat = DefaultPattern(
    initial_agent=business_analyst,
    agents=[business_analyst, coder, data_anayst, data_engineer],
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