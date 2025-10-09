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
Can you help me to find out the reasonable sale price of a house with following information:
1545,50,RM,50,6000,Pave,NA,Reg,Lvl,AllPub,Inside,Gtl,BrkSide,Norm,Norm,1Fam,1.5Fin,6,7,1939,1950,Gable,CompShg,MetalSd,MetalSd,None,0,TA,TA,BrkTil,TA,Gd,No,BLQ,452,LwQ,12,144,608,GasA,TA,Y,SBrkr,608,524,0,1132,1,0,1,0,2,1,TA,5,Typ,0,NA,Detchd,1939,Unf,1,240,TA,TA,Y,0,0,128,0,0,0,NA,MnPrv,NA,0,4,2010,WD,Abnorml?

    """,
    max_rounds=1000
)


#Can you help me to find out the reasonable sale price of a house with 3 bedroom, 2 bathroom, 1500 sqft, 1 car garage built in 2000?
#Could you help me identiy my house (with 3 bedrooms, 2 bathrooms, 1500 sqft, and a 1-car garage built in 2000) is in which segmentation? Such as luxury homes, affordable starter homes, or investment-ready properties...?
#Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.).

#Can you help me to find out the reasonable sale price of a house with following information: 1545,50,RM,50,6000,Pave,NA,Reg,Lvl,AllPub,Inside,Gtl,BrkSide,Norm,Norm,1Fam,1.5Fin,6,7,1939,1950,Gable,CompShg,MetalSd,MetalSd,None,0,TA,TA,BrkTil,TA,Gd,No,BLQ,452,LwQ,12,144,608,GasA,TA,Y,SBrkr,608,524,0,1132,1,0,1,0,2,1,TA,5,Typ,0,NA,Detchd,1939,Unf,1,240,TA,TA,Y,0,0,128,0,0,0,NA,MnPrv,NA,0,4,2010,WD,Abnorml?
