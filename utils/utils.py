import re
import streamlit as st
from .sidebar import ROLE_EMOJI
from autogen import OpenAIWrapper

def safe_md(text):
    return (
        text.replace("(", "\\(")
            .replace(")", "\\)")
            .replace("_", "\\_")
            .replace("+", "&#43;")
            .replace("~", "\\~")
            .replace("$", "\\$")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

config_list = [
    {
        "model": "gpt-4.1-nano",
        "api_type": "openai",
    }
]
client = OpenAIWrapper(config_list=config_list)

def convert_message_to_markdown(message):
    messages = [
        {"role": "user", "content": f"Convert this whole message to markdown format:\n{message}"}
    ]
    response = client.create(messages=messages)
    text = client.extract_text_or_completion_object(response)[0]
    if "```markdown" in text:
        try:
            return text.split("```markdown")[1].split("```")[0].strip()
        except IndexError:
            pass
    return text.strip()

# 2️⃣ Keep your chat renderer clean
def display_group_chat():
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar=ROLE_EMOJI.get(role, "")):
            st.markdown(f"**{role}**")
            if role in ("Coder", "System"):
                st.code(msg["content"])
            else:
                st.write(safe_md(msg["content"]))  # No safe_md needed anymore