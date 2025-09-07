import streamlit as st
from autogen import UserProxyAgent, LLMConfig, GroupChat, GroupChatManager, OpenAIWrapper
from multi_agents import (
    BusinessAnalyst, DataExplorer, DataEngineer, ModelBuilder, Evaluator, Manager, BusinessTranslator, CodeExecutor
)

def speaker_selection_method(last_speaker, group_chat):
    last_name = getattr(last_speaker, "name", None)
    if last_name == 'user':
        return group_chat.agent_by_name('')
    last_msg = ""
    if group_chat.messages and isinstance(group_chat.messages[-1], dict):
        last_msg = group_chat.messages[-1].get("content", "") or ""
    return manager.route_next(group_chat.agents, last_name, last_msg)

llm_config = LLMConfig.from_json(path="configs/llm_config.json")

user = UserProxyAgent(name="user", code_execution_config=False, human_input_mode="NEVER")

biz_analyst = BusinessAnalyst(llm_config)
data_explorer = DataExplorer(llm_config)
data_engineer = DataEngineer(llm_config)
model_builder = ModelBuilder(llm_config)
evaluator = Evaluator(llm_config)
biz_translator = BusinessTranslator(llm_config)
code_executor = CodeExecutor()

group_chat = GroupChat(
    agents=[user, biz_analyst, data_explorer, data_engineer, model_builder, evaluator, biz_translator, code_executor],
    max_round=10,
    speaker_selection_method=speaker_selection_method,
    messages=[]
)

manager = Manager(group_chat, llm_config)

task = task = """
    
"""

responses = user.run(manager, message=task)
# responses.process()

i=0

for event in responses.events:
    if hasattr(event.content, "content") \
        and hasattr(event.content, "sender") \
            and hasattr(event.content, "recipient"):
        print(event.content)
        print('-'*50)
        i+=1
        print(i)

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