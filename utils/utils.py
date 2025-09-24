from pathlib import Path
from autogen import OpenAIWrapper
from typing import Annotated
from autogen.agentchat.group import ContextVariables, ReplyResult, RevertToUserTarget, StayTarget
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor
from autogen.coding import CodeBlock

output_dir = Path("./artifacts")
output_dir.mkdir(parents=True, exist_ok=True)

server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
executor = JupyterCodeExecutor(server, output_dir=output_dir)

def run_code(code: Annotated[str, "Python code to run in Jupyter"]) -> ReplyResult:
    result = executor.execute_code_blocks(
        [CodeBlock(language="python", code=code)]
    )

    msg = f"Exit code: {result.exit_code}\n\nOutput:\n{result.output}\n\nStderr:\n{getattr(result, 'stderr', '')}"
    return ReplyResult(message=msg, target=StayTarget())

def custom_speaker_selection_func(last_speaker, group_chat):
    messages = group_chat.messages
    
    if last_speaker.name == 'user':
        return group_chat.agent_by_name('DataExplorer')

    # these states contains two steps, we will always call code_executor after the first step
    elif last_speaker.name in ['DataExplorer', 'DataEngineer', 'ModelBuilder']:
        return group_chat.agent_by_name('CodeExecutor')

    elif last_speaker.name == 'CodeExecutor':
        last_second_speaker_name = group_chat.messages[-2]["name"]

        if "exitcode: 1" in messages[-1]["content"]:
            return group_chat.agent_by_name(last_second_speaker_name)

        elif last_second_speaker_name == "DataExplorer":
            return group_chat.agent_by_name('DataEngineer')

        elif last_second_speaker_name == "DataEngineer":
            return group_chat.agent_by_name("ModelBuilder")
        
        elif last_second_speaker_name == "ModelBuilder":
            return None

def speaker_selection_method(last_speaker, group_chat):
    # last_speaker is an Agent or None; get its name safely
    last_name = getattr(last_speaker, "name", None)
    last_msg = ""
    if group_chat.messages and isinstance(group_chat.messages[-1], dict):
        last_msg = group_chat.messages[-1].get("content", "") or ""

    # manager is attached as group_chat.manager (created below)
    return group_chat.manager.route_next(group_chat.agents, last_name, last_msg)
    
def save_agent_code(chat_result):

    if "```python" in chat_result.chat_history[-1]["content"]:
        content = chat_result.chat_history[-1]["content"]
        content = content.split("```python")[1].split("```")[0].strip()
        with open("./agent_code/house_price_prediction.py", "w") as f:
            f.write(content)