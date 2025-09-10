import streamlit as st
from autogen import UserProxyAgent, LLMConfig, GroupChat, GroupChatManager, OpenAIWrapper
from multi_agents import (
    BusinessAnalyst, DataExplorer, DataEngineer, ModelBuilder, Evaluator, Manager, BusinessTranslator, CodeExecutor
)
from utils.utils import custom_speaker_selection_func

# 1. Load Config
llm_config = LLMConfig.from_json(path="configs/llm_config.json")

# 2. Initialise user agent
user = UserProxyAgent(
    name="user", 
    code_execution_config=False, 
    human_input_mode="NEVER"
)

data_explorer = DataExplorer(llm_config)
data_engineer = DataEngineer(llm_config)
model_builder = ModelBuilder(llm_config)
code_executor = CodeExecutor()

group_chat = GroupChat(
    agents=[user, data_explorer, data_engineer, model_builder, code_executor],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
    messages=[]
)

manager = GroupChatManager(groupchat=group_chat, llm_config=None)

task = task = """
    Please help me to build a model predict the sales price for each house.
    - The dataset is downloaded to this location: `./data/house_prices/house_prices_train.csv`.
    - All code will be executed in a Jupyter notebook, where previous states are saved.
"""

response = user.initiate_chat(manager, message=task)