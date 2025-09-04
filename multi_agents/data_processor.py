from autogen import ConversableAgent

class DataProcessor(ConversableAgent):
    def __init__(self, llm_config):
        super().__init__(
            name = 'Data_Processor',
            llm_config = llm_config,
            system_message = """
                You are the data processer. Given a dataset and a task, please write code to clean up the dataset. You goal is to prepare the dataset for model training.
                This includes but not limited to:
                1. handle missing values
                2. Remove unnecessary columns for model training
                3. Convert categorical variables to numerical variables
                4. Scale numerical variables
                5. other data preprocessing steps
                Please decide what data preprocessing steps are needed based on the data exploration results.
                When transforming data, try not to use `inplace=True`, but instead assign the transformed data to a new variable.
            """
        )