import streamlit as st
from autogen import UserProxyAgent, LLMConfig, GroupChat, GroupChatManager, OpenAIWrapper
from multi_agents import Manager, CodeExecutor, CodeSummarizer, DataExplorer, DataProcessor, ModelTrainer
from utils.utils import custom_speaker_selection_func, save_agent_code

llm_config = LLMConfig(
    api_type="openai", 
    model="gpt-4o-mini",
    temperature=0,
    timeout=10
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

responses = user.run(manager, message=task, max_turns=2)

for event in responses.events:
    if hasattr(event.content, "sender") and hasattr(event.content, "recipient") and hasattr(event.content, "content"):
        print("Sender: ", event.content.sender)
        print("Recipient: ", event.content.recipient)
        print("Content: ", event.content.content)
        print("-"*100)

# st.title("💸 Multi Agent Data Science Workflow Chat")

# # Input box for user query
# user_prompt = st.text_input("Please input your enquiry: ", task)

# if st.button("Run Agent"):
#     # Run the agent
#     responses = user.run(manager, message=task)

#     # Placeholder for live streaming
#     placeholder = st.empty()
#     buffer = ""

#     # Iterate through events as they arrive
#     for event in responses.events:
#         # Each event may be a message, tool call, or metadata
#         if hasattr(event.content, "content"):
#             # Append and update display
#             buffer += event.content.content
#             placeholder.markdown(buffer)

# save_agent_code(result)