from autogen import AssistantAgent, UserProxyAgent

# Create the assistant
markdown_agent = AssistantAgent(
    name="MarkdownFormatter",
    system_message=(
        "You are an assistant that formats plain text into Markdown format. "
        "Respond ONLY in Markdown syntax."
    ),
)

# Create a simulated user
user_agent = UserProxyAgent(name="user", code_execution_config=False)

# Input text
input_text = """
Meeting notes:
Discussed Q4 goals, approved new marketing budget, assigned tasks.
Alice: Design updates
Bob: Content strategy
Charlie: Ad campaigns
"""

# Run the agent and capture output
reply = markdown_agent.run(
    user_agent=user_agent,
    message=f"Format this text as Markdown:\n{input_text}"
)

# Print the Markdown result
print(reply.content)
