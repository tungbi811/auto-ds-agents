import streamlit as st
from utils.sidebar import Sidebar
from utils.utils import start_group_chat, display_group_chat

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

sidebar = Sidebar()
st.set_page_config(page_title="🤖 Multi-Agent for Data Science", layout="wide")
col1, col2, col3 = st.columns([0.25, 0.5, 0.25])
with col2:
    st.title("🤖 Multi-Agent for Data Science")
    st.write("👋 Upload your dataset and describe your requirements in the sidebar, then click **Run Analysis** to start.")

    # Run analysis when button is clicked
    if st.sidebar.button("🚀 Run Analysis", use_container_width=True, key="run_analysis"):
        if not sidebar.api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()
        if not sidebar.dataset_paths:
            st.warning("Please upload at least one dataset to proceed.")
            st.stop()
        if not sidebar.user_requirements.strip():
            st.warning("Please describe your data analysis requirements to proceed.")
            st.stop()

        st.session_state.messages.append(
            {"role": "User", "content": sidebar.user_requirements}
        )

        st.session_state.events = start_group_chat(
            provider_choice=sidebar.provider_choice,
            model_choice=sidebar.model_choice,
            api_key=sidebar.api_key,
            temperature=sidebar.temperature,
            dataset_paths=sidebar.dataset_paths,
            user_requirements=sidebar.user_requirements
        )

    if st.sidebar.button("🔄 Restart", use_container_width=True, key="restart"):
        del st.session_state.messages
        del st.session_state.events
        del st.session_state.awaiting_response
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
                        st.session_state.messages.append(
                            {
                                "role": st.session_state.event.content.sender, 
                                "content": "Calling tool: " + st.session_state.event.content.tool_calls[0].function.name
                            }
                        )
                    elif st.session_state.event.type == "tool_response":
                        st.session_state.messages.append(
                            {"role": "System", "content": st.session_state.event.content.content}
                        )
                    elif st.session_state.event.type == "input_request":
                        st.session_state.awaiting_response = True

                    elif st.session_state.event.type == "termination":
                        st.session_state.messages.append(
                            {"role": "System", "content": "The analysis has been completed."}
                        )
                        st.session_state.terminated = True
                        
                else:
                    st.session_state.event.content.respond(st.session_state.user_input)
                    st.session_state.user_input = ""

                st.rerun()
    else:
        st.info("The analysis has been completed. You can restart the process by clicking the 'Restart' button in the sidebar.")

    if st.session_state.awaiting_response:
        user_input = st.text_area("Your Response:", key="user_input")
        if st.button("Submit Response", key="submit_response"):
            if user_input.strip():
                st.session_state.awaiting_response = False
                st.rerun()
            else:
                st.warning("Please enter a response before submitting.")

