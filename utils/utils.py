def get_summary_prompt():
    try:
        return {
            1: """
                Summarize the content and format summary EXACTLY as follows:
                ---
                *Data set name*:
                `Acme Corp`
                ---
                *User problem*:
                `Regression, Classification or Time-series`
                ---
                *Target variable*
                `write here the target variable`
                ---
                *Indications on how to write the code to read data*
                `Write the code that you read the data`
                """,
            2: """
                Summarize the content and format summary EXACTLY as follows:
                ---
                *Problem*:
                `Write the machine learning problem`
                ---
                *Correlations*:
                `Relevant correlations`
                ---
                *Columns name*:
                `Columns: column1, column2...`
                ---               
                *Relevant insights*:
                `Useful insights found for the next agents`
                ---
                """,
            3: """
                Summarize the content and format summary EXACTLY as follows:
                ---
                *Problem*:
                `problem written here`
                ---
                *Machine learning model to use*:
                `ML model`
                ---
                *Explanation and alternatives*:
                `Explanation of why that model was chosen`
                ---
                """,
            4: """
                Summarize the content and format summary EXACTLY as follows:
                ---
                *Transformations*:
                `Transformations you've done to the data`
                ---
                *Splitting*:
                `The split you've done and where you save the data`
                ---
                **Read the data**
                `pd.read_csv('X_train.csv')`
                `pd.read_csv('y_train.csv')`
                `pd.read_csv('X_test.csv')`
                `pd.read_csv('y_test.csv')`
                """,
            5: """
                Summarize the content and format summary EXACTLY as follows:
                ---
                *ML model used*:
                `ML model`
                ---
                *Place where you saved predictions*:
                `acmecorp.com`
                ---
                *Results of the evaluations*:
                `Metric: result`
                ---
                """,
            6: """
                Summarize the content and format summary EXACTLY as follows:
                ---
                *Data set name*:
                `name of the dataset`
                ---
                *User problem*:
                `problem of the user`
                ---
                *Target variable*
                `the variable we're trying to predict`
                ---    
                **Correlations:**
                'Correlations found'
                ---
                **Columns:**
                'the columns'    
                ---
                **Relevant Insights:**
                'The relevant insights'
                ---
                *Machine learning model to use*:
                Random Forest Regressor
                ---
                *Explanation and alternatives*:
                'Explanations of the machine learning model used'
                ---
                *Transformations*:
                `the transformations made to the data`
                ---
                *Splitting*:
                `The splitting done to the data`
                ---
                *Results of the evaluations*:
                `results of the metrics`
                ---
                """
        }
    except Exception as e:
        logging.error("Error in _get_summary_prompt", exc_info=True)
        raise