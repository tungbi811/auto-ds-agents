from multi_agents import BusinessAnalyst, Coder, DataExplorer, DataEngineer, Modeler, Evaluator
from autogen import UserProxyAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables
from autogen.agentchat.group.patterns import DefaultPattern

context_variables = ContextVariables(
    data={
        "current_agent": None,
        "goal": None,
        "problem_type": None,
        "key_metrics": None
    }
)

biz_analyst = BusinessAnalyst()
coder = Coder()
data_explorer = DataExplorer()
data_engineer = DataEngineer()

user = UserProxyAgent(
    name="User",
    code_execution_config=False
)

biz_analyst.handoffs.set_after_work(AgentTarget(data_explorer))
data_explorer.handoffs.set_after_work(AgentTarget(data_engineer))

group_chat = DefaultPattern(
    initial_agent=biz_analyst,
    agents=[biz_analyst],
    user_agent=user,
    context_variables=context_variables
)

initiate_group_chat(
    pattern=group_chat,
    messages="""
        Here is the dataset path: ./data/house_prices/train.csv.
        Here is the test set path: ./data/house_prices/test.csv.
        Here is the sample_submission path: ./data/house_prices/sample_submission.csv.
        Export for me a clean CSV file with predictions for the test set, ready for submission to Kaggle.
    """,
    max_rounds=100
)