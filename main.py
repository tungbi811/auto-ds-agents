from multi_agents import BusinessAnalyst, DataAnalyst, DataEngineer, DataScientist, BusinessTranslator, Coder
from autogen import UserProxyAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables
from autogen.agentchat.group.patterns import DefaultPattern

context_variables = ContextVariables(data={
    "current_agent": "",
    "objective": "",
    "problem_type": "",
    "stakeholders_expectations": [],
    "research_questions": [],
})

coder = Coder()
business_analyst = BusinessAnalyst()
data_analyst = DataAnalyst()
data_engineer = DataEngineer()
data_scientist = DataScientist()
business_translator = BusinessTranslator()

user = UserProxyAgent(
    name="User",
    code_execution_config=False
)

business_analyst.handoffs.set_after_work(AgentTarget(data_analyst))
data_analyst.handoffs.set_after_work(AgentTarget(data_engineer))
data_engineer.handoffs.set_after_work(AgentTarget(data_scientist))
data_scientist.handoffs.set_after_work(AgentTarget(business_translator))

group_chat = DefaultPattern(
    initial_agent=business_analyst,
    agents=[business_analyst, coder, data_analyst, data_engineer, data_scientist, business_translator],
    user_agent=user,
    context_variables=context_variables
)

initiate_group_chat(
    pattern=group_chat,
    messages="""
        Here is the dataset path: ./data/house_prices/train.csv. 
        Can you help me to find out the reasonable sale price of a house with 3 bedroom, 2 bathroom, 1500 sqft, 1 car garage built in 2000?
    """,
    max_rounds=150
)