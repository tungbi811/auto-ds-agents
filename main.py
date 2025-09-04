from autogen import UserProxyAgent, LLMConfig, GroupChat, GroupChatManager, OpenAIWrapper
from multi_agents import Manager, CodeExecutor, CodeSummarizer, DataExplorer, DataProcessor, ModelTrainer
from utils.utils import custom_speaker_selection_func, save_agent_code

llm_config = LLMConfig(
    api_type="openai", 
    model="gpt-4o-mini",
    temperature=0,
    timeout=120
)

user = UserProxyAgent(name="user", code_execution_config=False)
data_explorer = DataExplorer(llm_config)
data_processor = DataProcessor(llm_config)
model_trainer = ModelTrainer(llm_config)
code_summarizer = CodeSummarizer(llm_config)
code_executor = CodeExecutor()

group_chat = GroupChat(
    agents=[user, data_explorer, data_processor, model_trainer, code_executor, code_summarizer],
    max_round=20,
    speaker_selection_method=custom_speaker_selection_func,
    messages=[]
)

manager = GroupChatManager(group_chat, llm_config=None)

task = """
    Please help me to build a model predict the sales price for each house.
    - The dataset is downloaded to this location: `./data/house_prices/house_prices_train.csv`.
    - All code will be executed in a Jupyter notebook, where previous states are saved.
"""

result = user.initiate_chat(
    manager,
    message=task,
)

save_agent_code(result)

print("Hello World")