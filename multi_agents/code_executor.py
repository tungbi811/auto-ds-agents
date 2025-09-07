from pathlib import Path
from autogen import ConversableAgent
from autogen.coding.jupyter import LocalJupyterServer, JupyterCodeExecutor

class CodeExecutor(ConversableAgent):
    def __init__(self):
        output_dir = Path("./artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)

        server = LocalJupyterServer(log_file='./logs/jupyter_gateway.log')
        executor = JupyterCodeExecutor(server, output_dir=output_dir)

        super().__init__(
            name="CodeExecutor",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config={"executor": executor},
        )
