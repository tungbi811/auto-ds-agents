import os
import json
import streamlit as st
from multi_agents.group_chat import GroupChat
from utils.sidebar import Sidebar
from utils.utils import display_group_chat

if "messages" not in st.session_state:
    st.session_state.messages = []
if "events" not in st.session_state:
    st.session_state.events = None
if "event" not in st.session_state:
    st.session_state.event = None
if "awaiting_response" not in st.session_state:
    st.session_state.awaiting_response = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "terminated" not in st.session_state:
    st.session_state.terminated = False
if "last_agent_name" not in st.session_state:
    st.session_state.last_agent_name = None

sidebar = Sidebar()
st.set_page_config(page_title="🤖 Multi-Agent for Data Science", layout="wide")
col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
with col2:
    st.title("🤖 Multi-Agent for Data Science")
    st.write("👋 Upload your dataset and describe your requirements in the sidebar, then click **Run Analysis** to start.")

    # Run analysis when button is clicked
    if st.sidebar.button("🚀 Run Analysis", use_container_width=True, key="run_analysis"):
        if not sidebar.api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = sidebar.api_key
        if not sidebar.dataset_paths:
            st.warning("Please upload at least one dataset to proceed.")
            st.stop()
        if not sidebar.user_requirements.strip():
            st.warning("Please describe your data analysis requirements to proceed.")
            st.stop()

        st.session_state.messages.append(
            {"role": "User", "content": sidebar.user_requirements}
        )

        group_chat = GroupChat()

        st.session_state.events = group_chat.run(
            dataset_paths=sidebar.dataset_paths,
            user_requirements=sidebar.user_requirements
        )

    if st.sidebar.button("🔄 Restart", use_container_width=True, key="restart"):
        del st.session_state.messages
        del st.session_state.events
        del st.session_state.event
        del st.session_state.awaiting_response
        del st.session_state.user_input
        del st.session_state.terminated
        del st.session_state.last_agent_name
        st.rerun()

    display_group_chat()
    if not st.session_state.terminated:
        if not st.session_state.awaiting_response: 
            if st.session_state.events:
                if not st.session_state.user_input:
                    st.session_state.event = next(st.session_state.events)

                    if st.session_state.event.type == "text":
                        sender = st.session_state.event.content.sender
                        message = st.session_state.event.content.content
                        if not(sender == "User" and sidebar.user_requirements.strip() in message):
                            st.session_state.messages.append(
                                {"role": sender, "content": message}
                            )
                        
                    elif st.session_state.event.type == "tool_call":
                        if st.session_state.event.content.sender == "Coder":
                            st.session_state.messages.append(
                                {
                                    "role": "Coder", 
                                    "content": json.loads(st.session_state.event.content.tool_calls[0].function.arguments)["code"]
                                }
                            )
                        else:
                            st.session_state.last_agent_name = st.session_state.event.content.sender
                    elif st.session_state.event.type == "tool_response":
                        if st.session_state.last_agent_name:
                            st.session_state.messages.append(
                                {
                                    "role": st.session_state.last_agent_name, 
                                    "content": st.session_state.event.content.content
                                }
                            )
                            st.session_state.last_agent_name = None
                        else:
                            st.session_state.messages.append(
                                {
                                    "role": "System", 
                                    "content": st.session_state.event.content.content
                                }
                            )

                    elif st.session_state.event.type == "input_request":
                        st.session_state.awaiting_response = True
                    elif st.session_state.event.type == "termination":
                        st.session_state.messages.append(
                            {"role": "System", "content": st.session_state.event.content.termination_reason}
                        )
                        st.session_state.terminated = True
                        
                else:
                    st.session_state.event.content.respond(st.session_state.user_input)
                    st.session_state.user_input = ""

                st.rerun()
    else:
        st.info("The analysis has been completed. You can restart the process by clicking the 'Restart' button in the sidebar.")

    if st.session_state.awaiting_response:
        user_input = st.text_area("Replying as User. Provide feedback to Business Analyst. Type 'exit' to end the conversation:", key="user_input")
        if st.button("Submit Response", key="submit_response"):
            if user_input.strip():
                st.session_state.awaiting_response = False
                st.rerun()
            else:
                st.warning("Please enter a response before submitting.")

# Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.)?
# Predict the sale price of a house based on location, size, features (bedrooms, bathrooms, parking), and historical market data.