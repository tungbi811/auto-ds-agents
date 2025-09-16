from autogen import UserProxyAgent, GroupChat, GroupChatManager
from multi_agents import ChatManager, ProjectManager, BusinessAnalyst

bis_analyst = BusinessAnalyst()
project_manager = ProjectManager()
user = UserProxyAgent(
    name="User", 
    code_execution_config=False,
    human_input_mode="ALWAYS"
)
group_chat = GroupChat(
    agents=[user, project_manager, bis_analyst],
    max_round=20,
    speaker_selection_method="auto"
)
chat_manager = ChatManager(group_chat=group_chat)

chat_result = user.initiate_chat(
    chat_manager, 
    message="""
        Please help me to build a model predict the sales price for each house.
        The dataset is downloaded to this location: `./data/house_prices/train.csv.
    """
)
