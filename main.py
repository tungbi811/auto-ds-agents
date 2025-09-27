from multi_agents import User, BusinessAnalyst, DataExplorer, DataEngineer, ModelConsulter, Modeller, Reporter
from utils.utils import get_summary_prompt
from autogen.agentchat import initiate_group_chat

dataset_path = "./data/house_prices/train.csv"
user_requirement = "I want to predict house prices"

user = User()
biz_analyst = BusinessAnalyst()
data_explorer = DataExplorer()
data_engineer = DataEngineer()
model_consulter = ModelConsulter()
modeller = Modeller()
reporter  = Reporter()

carryover_msg = f"The needed data is in {dataset_path}"
chat_configurations = [
    {
        "summary_method": "reflection_with_llm",
        **({"carryover": carryover_msg} if i < 4 else {})
    }
    for i in range(7)
]

task_prompts = [
    f"This is my requirement: {user_requirement}. What is the Machine learning problem?",
    "I want to perform data analysis",
    "Based on the relevant insights, identify the most relevant machine learning model to use.",
    "Base on the relevant insights, make the necessary transformations to the data, separate the data by features and target based on the problem type and split the dataset.",
    "Fit the training data, make predictions, and evaluate them.",
    "Compile a detailed report with insights from other agents.",
]

user.initiate_chats(
    agents=[biz_analyst, data_explorer, data_engineer, modeller, model_consulter, reporter],

)