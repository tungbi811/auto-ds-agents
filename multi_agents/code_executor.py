from pathlib import Path
from autogen import UserProxyAgent
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

class CodeExecutor(UserProxyAgent):
    def __init__(self):
        log_path = Path("../logs/jupyter_gateway.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        output_dir = Path("../artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        server = LocalJupyterServer(log_file=str(log_path))
        executor = JupyterCodeExecutor(server, output_dir=output_dir)

        super().__init__(
            name="CodeExecutor",
            human_input_mode="NEVER",
            system_message=(
                "You are a Code Executor. Execute the single fenced code block you receive and "
                "return stdout/stderr and any printed paths to artifacts. "
                "If no runnable code block is present, respond with an error message."
            ),
            human_input_mode="NEVER",
            code_execution_config={
                "executor": executor,         # REQUIRED to actually run the code
                "use_docker": False,          # set True if you want container isolation (needs setup)
                "last_n_messages": 1,         # only consider the latest message for execution
                "quiet": True,                # suppress extra chatter
            },
        )