from pathlib import Path
from autogen import UserProxyAgent
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

class CodeExecutor(UserProxyAgent):
    def __init__(self):
        output_dir = Path("./artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
        executor = JupyterCodeExecutor(server, output_dir=output_dir)

        super().__init__(
            name="CodeExecutor",
            human_input_mode="NEVER",
            system_message=("""
                You are the Code Executor Agent.
                Your job is to run code provided by other agents.

                Rules:
                - Accept exactly one fenced code block per request.
                - Prefer Python; bash is allowed only for simple file operations.
                - Run the code in a persistent Jupyter kernel.
                - Save all artifacts to ./artifacts.
                - Return stdout, stderr, and any printed file paths.
                """
            ),
            code_execution_config={
                "executor": executor,         # REQUIRED to actually run the code
                "last_n_messages": 1,         # only consider the latest message for execution
                "quiet": True,                # suppress extra chatter
            },
        )
