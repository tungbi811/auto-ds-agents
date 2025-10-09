from multi_agents import BusinessAnalyst, DataAnalyst, DataEngineer, DataScientist, BusinessTranslator, Coder
from autogen import UserProxyAgent
from autogen.agentchat import initiate_group_chat, run_group_chat
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

response = run_group_chat(
    pattern=group_chat,
    messages="""
        Here is the dataset path: ./data/house_prices/train.csv. 
        Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.).
    """,
    max_rounds=10
)

for event in response.events:
    print(event)

#Can you help me to find out the reasonable sale price of a house with 3 bedroom, 2 bathroom, 1500 sqft, 1 car garage built in 2000?
#Could you help me identiy my house (with 3 bedrooms, 2 bathrooms, 1500 sqft, and a 1-car garage built in 2000) is in which segmentation? Such as luxury homes, affordable starter homes, or investment-ready properties...?
#Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.).