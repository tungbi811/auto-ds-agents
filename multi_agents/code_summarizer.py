from autogen import ConversableAgent

class CodeSummarizer(ConversableAgent):
    def __init__(self, llm_config):
        super().__init__(
            name = 'Code_Summarizer',
            llm_config = llm_config,
            system_message = """
                You are the code summarizer. Given a machine learning task and previous code snippets, please integrate all error-free code into a single code snippet.
                Please also provide a brief summary of the data exploration, data processing, and model training steps, and conclude what model is the best for the task.
                You should give the full code to reproduce the data exploration, data processing, and model training steps, and show the results with different metrics.
            """
        )