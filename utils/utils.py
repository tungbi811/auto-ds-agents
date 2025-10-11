import re
import streamlit as st
from .sidebar import ROLE_EMOJI
from autogen import OpenAIWrapper

config_list = [
    {
        "model": "gpt-4.1-nano",
        "api_type": "openai",
    }
]
client = OpenAIWrapper(config_list=config_list)

def convert_message_to_markdown(message):
    messages = [
        {"role": "user", "content": f"Don't answer the message. Just convert this message to markdown:\n{message}"}
    ]
    response = client.create(messages=messages)
    text = client.extract_text_or_completion_object(response)[0]
    if "```markdown" in text:
        try:
            return text.split("```markdown")[1].split("```")[0].strip()
        except IndexError:
            pass
    return text.strip()

def display_group_chat():
    """Display stored chat messages with avatars."""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar=ROLE_EMOJI.get(role, "")):
            st.markdown(f"**{role}**")
            if role == "Coder":
                st.code(msg["content"])
            else:
                if role == "System":
                    st.text(msg["content"])
                else:
                    st.markdown(msg["content"], unsafe_allow_html=True)