import streamlit as st
from autogen import OpenAIWrapper

ROLE_EMOJI = {
    "User": "🧑‍💻",
    "BusinessAnalyst": "💼",
    "BusinessTranslator": "🗣️",
    "DataAnalyst": "🔎",
    "DataEngineer": "🛠️",
    "DataScientist": "📊",
    "Coder": "🧠",
    "Assistant": "🤖",
    "System": "⚙️"
}

config_list = [
    {
        "model": "gpt-4.1-nano",
        "api_type": "openai",
    }
]
client = OpenAIWrapper(config_list=config_list)

def convert_message_to_markdown(message):
    messages = [
        {"role": "user", "content": f"Convert the whole message to markdown format (do not summarise, just convert):\n{message}"}
    ]
    response = client.create(messages=messages)
    text = client.extract_text_or_completion_object(response)[0]
    if "```markdown" in text:
        try:
            return text.split("```markdown")[1].split("```")[0].strip()
        except IndexError:
            pass
    return text.strip()


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

def display_group_chat():
    expander_buffer = []  # temporary buffer for consecutive expander messages

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        in_expander = msg.get("in_expander", False)

        if in_expander:
            # Add message to current expander buffer
            expander_buffer.append(msg)
        else:
            # If we hit a normal message and there are buffered expander messages, render them first
            if expander_buffer:
                with st.expander("💡 Detailed Response", expanded=False):
                    for emsg in expander_buffer:
                        erole = emsg["role"]
                        econtent = emsg["content"]
                        with st.chat_message(erole, avatar=ROLE_EMOJI.get(erole, "")):
                            st.markdown(f"**{erole}**")
                            if erole in ["Coder", "System"]:
                                st.code(econtent)
                            else:
                                if "```markdown" in econtent:
                                    st.write(safe_md(econtent.split("```markdown")[1].split("```")[0].strip()))
                                else:
                                    st.write(safe_md(econtent))
                expander_buffer = []  # reset buffer

            # Render this normal message
            with st.chat_message(role, avatar=ROLE_EMOJI.get(role, "")):
                st.markdown(f"**{role}**")
                if "```markdown" in content:
                    st.write(safe_md(content.split("```markdown")[1].split("```")[0].strip()))
                else:
                    st.write(safe_md(content))

    if expander_buffer:
        with st.expander("🧠 Thinking ...", expanded=False):
            for emsg in expander_buffer:
                erole = emsg["role"]
                econtent = emsg["content"]
                with st.chat_message(erole, avatar=ROLE_EMOJI.get(erole, "")):
                    st.markdown(f"**{erole}**")
                    if erole in ["Coder", "System"]:
                        st.code(econtent)
                    else:
                        if "```markdown" in econtent:
                            st.write(safe_md(econtent.split("```markdown")[1].split("```")[0].strip()))
                        else:
                            st.write(safe_md(econtent))
