from crew.crew_tools import ExecutePythonCodeTool, set_session_id
import uuid

def test_code_execution():
    """Test the Execute Python Code tool"""
    
    # Set a test session
    session_id = str(uuid.uuid4())
    set_session_id(session_id)
    
    # Create tool instance
    code_tool = ExecutePythonCodeTool()
    
    # Test basic code execution
    test_code = """
import pandas as pd
import numpy as np
print("Testing code execution...")
print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)

# Test data creation
test_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
df = pd.DataFrame(test_data)
print("DataFrame created:")
print(df)
print("Shape:", df.shape)
    """
    
    print("=== Testing Python Code Execution ===")
    print("Session ID:", session_id)
    print("Code to execute:")
    print(test_code)
    print("=== Execution Result ===")

    # Execute the code
    result = code_tool._run(test_code)
    print(result)
    
    return result

if __name__ == "__main__":
    test_code_execution()