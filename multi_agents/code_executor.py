from pathlib import Path
from autogen import UserProxyAgent
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

class CodeExecutor(UserProxyAgent):
    def __init__(self):
        log_path = Path("../logs/jupyter_gateway.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        server = LocalJupyterServer(log_file=str(log_path))
        
        output_dir = Path("..") / "agent_code"
        output_dir.mkdir(exist_ok=True)  

        super().__init__(
            name="Code_Executor",
            system_message="Executor. Execute the code written by the Coder and report the result.",
            human_input_mode="NEVER",
            code_execution_config={
                "executor": JupyterCodeExecutor(server, output_dir=output_dir)
            }
        )