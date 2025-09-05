import streamlit as st
from autogen import ConversableAgent, LLMConfig

# Configure the LLM
llm_config = LLMConfig(
    api_type="openai", 
    model="gpt-4o-mini",
    temperature=0,
    timeout=120,
    stream=True
)

# Create a financial agent
finance_agent = ConversableAgent(
    name="finance_agent",
    system_message="You are a financial assistant who helps analyze financial data and transactions.",
    llm_config=llm_config,
)

st.title("💸 Financial Agent Chat")

# Input box for user query
user_prompt = st.text_input("Ask the financial agent something:", 
                            "Can you explain what makes a transaction suspicious?")

if st.button("Run Agent"):
    # Run the agent
    responses = finance_agent.run(message=user_prompt, max_turns=1)

    # Placeholder for live streaming
    placeholder = st.empty()
    buffer = ""

    # Iterate through events as they arrive
    for event in responses.events:
        # Each event may be a message, tool call, or metadata
        if hasattr(event.content, "content"):
            # Append and update display
            buffer += event.content.content
            placeholder.markdown(buffer)
