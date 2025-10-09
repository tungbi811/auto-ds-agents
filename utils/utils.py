import streamlit as st
from .sidebar import ROLE_EMOJI
from autogen.agentchat import run_group_chat
from autogen import UserProxyAgent, LLMConfig
from autogen.agentchat.group.patterns import DefaultPattern
from autogen.agentchat.group import AgentTarget, ContextVariables
from multi_agents import BusinessAnalyst, DataAnalyst, DataEngineer, DataScientist, Coder, BusinessTranslator

def start_group_chat(dataset_paths, user_requirements):
    context_variables = ContextVariables(data={
        "current_agent": "",
        "objective": "",
        "problem_type": "",
        "stakeholders_expectations": [],
        "research_questions": [],
    })
    
    business_analyst = BusinessAnalyst()
    # data_analyst = DataAnalyst()
    data_engineer = DataEngineer()
    data_scientist = DataScientist()
    business_translator = BusinessTranslator()
    coder = Coder()
    user = UserProxyAgent(name="User", code_execution_config=False)

    business_analyst.handoffs.set_after_work(AgentTarget(data_engineer))
    # data_analyst.handoffs.set_after_work(AgentTarget(data_engineer))
    data_engineer.handoffs.set_after_work(AgentTarget(data_scientist))
    data_scientist.handoffs.set_after_work(AgentTarget(business_translator))

    pattern = DefaultPattern(
        initial_agent=business_analyst,
        agents=[business_analyst, coder, data_engineer, data_scientist, business_translator],
        user_agent=user,
        context_variables=context_variables
    )

    message = f"""
        Data path: {dataset_paths}
        Requirements: {user_requirements}
    """

    response = run_group_chat(
        pattern=pattern,
        messages=message,
        max_rounds=150
    )

    return response.events

def display_group_chat():
    """Display stored chat messages with avatars."""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar=ROLE_EMOJI.get(role, "")):
            st.markdown(f"**{role}**")
            if role == "Coder":
                st.code(msg["content"])
            else:
                st.text(msg["content"])