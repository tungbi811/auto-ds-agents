import json
import autogen

config_list = autogen.LLMConfig(
    cache_seed=41,
    api_type = 'openai',
    model = 'gpt-4o-mini',
    temperature = 0
)

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=config_list
)
# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="ALWAYS",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

math_problem_to_solve = """
Find $a + b + c$, given that $x+y \\neq -1$ and
\\begin{align}
    ax + by + c & = x + 7,\\
    a + bx + cy & = 2x + 6y,\\
    ay + b + cx & = 4x + y.
\\end{align}.
"""

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(assistant, message=math_problem_to_solve)

print(user_proxy.chat_messages[assistant])
json.dump(user_proxy.chat_messages[assistant], open("conversations.json", "w"), indent=2)  # noqa: SIM115