from autogen import UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget
from multi_agents import  Manager, Planner, CodeWriter, CodeExecutor, Summariser, Evaluator
from autogen.agentchat.group import OnCondition, StringLLMCondition

manager = Manager()
planner = Planner()
code_writer = CodeWriter()
code_executor = CodeExecutor()
summariser = Summariser()
evaluator = Evaluator()

manager.handoffs.set_after_work(AgentTarget(planner))
planner.handoffs.set_after_work(AgentTarget(code_writer))
code_executor.handoffs.set_after_work(AgentTarget(summariser))
summariser.handoffs.add_llm_conditions([
    OnCondition(
        target=AgentTarget(code_writer),
        condition=StringLLMCondition(prompt="If code execution failed, return to code writer and explain reason")
    ),
    OnCondition(
        target=AgentTarget(evaluator),
        condition=StringLLMCondition(prompt="If code execution is successful, summary results and route to evaluator ")
    )
])
evaluator.handoffs.set_after_work(AgentTarget(code_writer))

user = UserProxyAgent(
    name="User", 
    code_execution_config=False,
    human_input_mode="ALWAYS"
)

pattern = DefaultPattern(
    initial_agent=manager,
    agents=[manager, planner, code_writer, code_executor, summariser, evaluator],
    user_agent=user
)

result, final_context, last_agent = initiate_group_chat(
    pattern=pattern,
    messages="""
        Please help me to build a model predict the sales price for each house.
        The dataset is downloaded to this location: `./data/house_prices/train.csv.
    """,
    max_rounds=20
)

