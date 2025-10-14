import os
import json
import streamlit as st
from multi_agents.group_chat import GroupChat
from utils.sidebar import Sidebar
# from utils.utils import display_group_chat  # replaced by render_messages_messenger_strict()

# -------------------------------
# Streamlit page setup
# -------------------------------
st.set_page_config(page_title="🤖 Multi-Agent for Data Science", layout="wide")

# -------------------------------
# Helper: Messenger-style rendering (strict visibility)
# -------------------------------
def render_messages_messenger_strict(
    bt_roles=("BusinessTranslator", "Business Translator"),
    ba_roles=("BusinessAnalyst", "Business Analyst"),
    user_role="User",
    code_roles=("Coder",),
    role_avatars=None,
):
    """
    Messenger-like UI with strict visibility rules:
      - User messages: always visible (right-aligned).
      - Business Translator (BT): keep the LAST BT message of EACH user turn visible
        (i.e., the last BT between two User messages, and the last BT at the tail).
        All other BT messages go to backchannel.
      - Business Analyst (BA): visible ONLY if the message looks like a question (heuristic).
      - All other messages go into a hidden 'backchannel' expander.
    """
    if role_avatars is None:
        role_avatars = {
            "User": "🧑",
            "BusinessTranslator": "🧭",
            "Business Translator": "🧭",
            "BusinessAnalyst": "📊",
            "Business Analyst": "📊",
            "DataExplorer": "🔎",
            "DataEngineer": "🧱",
            "Modeler": "📈",
            "Coder": "💻",
            "System": "⚙️",
        }

    def is_question(text: str) -> bool:
        """Simple heuristics to detect questions for BA visibility."""
        if not text:
            return False
        t = text.strip()
        if "?" in t:
            return True
        prefixes = (
            "q:", "question:", "ask:",
            "can you", "could you", "would you", "please provide",
            "why ", "how ", "what ", "which ", "when ", "where "
        )
        return t.lower().startswith(prefixes)

    msgs = st.session_state.get("messages", [])
    if not msgs:
        st.info("No conversation yet. Start an analysis from the sidebar.")
        return

    # --- NEW: Determine which BT messages are the last within each user turn ---
    # We scan messages and whenever we hit a User message (and once at the end),
    # we mark the most recent BT since the previous User as visible.
    visible_bt_indices = set()
    start = 0
    # Append a sentinel User at the end to close the last turn
    for j, m in enumerate(msgs + [{"role": user_role, "content": ""}]):
        if m.get("role") == user_role:
            # Look back from j-1 to start, keep the last BT
            for k in range(j - 1, start - 1, -1):
                if msgs[k].get("role") in bt_roles:
                    visible_bt_indices.add(k)
                    break
            start = j + 1  # next segment starts after this user

    main_thread = []   # messages to show inline (Messenger style)
    backchannel = []   # messages hidden under expander

    # Partition messages according to rules while preserving chronological order
    for idx, m in enumerate(msgs):
        role = m.get("role", "")
        content = m.get("content", "")

        if role == user_role:
            main_thread.append(m)
            continue

        if role in bt_roles:
            # NEW: show only the last BT per user turn
            if idx in visible_bt_indices:
                main_thread.append(m)
            else:
                backchannel.append(m)
            continue

        if role in ba_roles:
            # Show BA only when it appears to be a question
            if is_question(content):
                main_thread.append(m)
            else:
                backchannel.append(m)
            continue

        # Everything else goes to backchannel
        backchannel.append(m)

    # ---- MAIN THREAD (Messenger style) ----
    st.markdown("### Conversation")
    for m in main_thread:
        role = m.get("role", "")
        content = m.get("content", "")
        avatar = role_avatars.get(role, "🤖")
        chat_side = "user" if role == user_role else "assistant"

        with st.chat_message(chat_side, avatar=avatar):
            if role != user_role:
                st.markdown(f"**{role}**")
            is_code_like = (
                (role in code_roles)
                or content.lstrip().startswith(("def ", "class ", "import ", "from "))
                or content.lstrip().startswith(("{", "["))
            )
            if is_code_like and role != user_role:
                st.code(content, language="python")
            else:
                st.markdown(content)

    # ---- BACKCHANNEL (hidden, but chat-styled) ----
    with st.expander(f"🗂️ Agent backchannel (hidden) — {len(backchannel)} message(s)", expanded=False):
        if not backchannel:
            st.caption("No hidden agent messages.")
        else:
            # Group by role to make it easier to scan
            grouped = {}
            for m in backchannel:
                grouped.setdefault(m.get("role", "Unknown"), []).append(m)

            for role, items in grouped.items():
                with st.expander(f"{role} ({len(items)})", expanded=False):
                    for i, m in enumerate(items, 1):
                        content = m.get("content", "")
                        avatar = role_avatars.get(role, "🤖")
                        with st.chat_message("assistant", avatar=avatar):
                            st.markdown(f"**{role} · message #{i}**")
                            is_code_like = (
                                (role in code_roles)
                                or content.lstrip().startswith(("def ", "class ", "import ", "from "))
                                or content.lstrip().startswith(("{", "["))
                            )
                            if is_code_like:
                                st.code(content, language="python")
                            else:
                                st.markdown(content)



# -------------------------------
# Initialize session state (unchanged)
# -------------------------------
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

# -------------------------------
# Main layout (unchanged logic)
# -------------------------------
sidebar = Sidebar()
col1, col2, col3 = st.columns([0.05, 0.9, 0.05])
with col2:
    st.title("🤖 Multi-Agent for Data Science")
    st.write("👋 Upload your dataset and describe your requirements in the sidebar, then click **Run Analysis** to begin.")

    # --- Run analysis button ---
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

    # --- Restart button ---
    if st.sidebar.button("🔄 Restart", use_container_width=True, key="restart"):
        for k in ["messages", "events", "event", "awaiting_response", "user_input", "terminated", "last_agent_name"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    # --- Render messages (Messenger style, strict) ---
    render_messages_messenger_strict()

    # --- Event loop handling (unchanged) ---
    if not st.session_state.terminated:
        if not st.session_state.awaiting_response:
            if st.session_state.events:
                if not st.session_state.user_input:
                    st.session_state.event = next(st.session_state.events)

                    if st.session_state.event.type == "text":
                        sender = st.session_state.event.content.sender
                        message = st.session_state.event.content.content
                        if not (sender == "User" and sidebar.user_requirements.strip() in message):
                            st.session_state.messages.append(
                                {"role": sender, "content": message}
                            )

                    elif st.session_state.event.type == "tool_call":
                        if st.session_state.event.content.sender == "Coder":
                            st.session_state.messages.append(
                                {
                                    "role": "Coder",
                                    "content": json.loads(
                                        st.session_state.event.content.tool_calls[0].function.arguments
                                    )["code"]
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
                                {"role": "System", "content": st.session_state.event.content.content}
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
        st.info("✅ The analysis is complete. You can restart it using the **Restart** button in the sidebar.")

    if st.session_state.awaiting_response:
        user_input = st.text_area("Replying as User. Type 'exit' to end the conversation:", key="user_input")
        if st.button("Submit Response", key="submit_response"):
            if user_input.strip():
                st.session_state.awaiting_response = False
                st.rerun()
            else:
                st.warning("Please enter a response before submitting.")

# Example prompts:
# Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.)?
# Predict the sale price of a house based on location, size, features (bedrooms, bathrooms, parking), and historical market data.
