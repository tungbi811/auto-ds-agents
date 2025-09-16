# test_persistence.py
from crew.crew_tools import toolkit, set_session_id

# Test 1: Basic execution
set_session_id("test_session")

code1 = """
print("Test 1: Basic print")
x = 10
print(f"Variable x = {x}")
"""

result1 = toolkit.execute_python_code(code1)
print("Result 1:", result1)

# Test 2: Check persistence
code2 = """
print(f"Test 2: Can we see x? {x}")
y = 20
print(f"New variable y = {y}")
"""

result2 = toolkit.execute_python_code(code2)
print("Result 2:", result2)

# Test 3: Dataset loading
code3 = """
import glob
datasets = glob.glob('workspace/*.csv')
print(f"Found datasets: {datasets}")
"""

result3 = toolkit.execute_python_code(code3)
print("Result 3:", result3)






from crew.crew_tools import toolkit, set_session_id

set_session_id("persist_test")

# First execution
code1 = "x = 100nprint(f'Set x to {x}')"
result1 = toolkit.execute_python_code(code1)
print("Result 1:", result1)

# Check script content
with open("workspace/default_code_persist_test.py", "r") as f:
    content1 = f.read()
print("Script after code1:", "x = 100" in content1)

# Second execution  
code2 = "y = 200nprint(f'Set y to {y}')nprint(f'x is {x}')"
result2 = toolkit.execute_python_code(code2)
print("Result 2:", result2)

# Check script content again
with open("workspace/default_code_persist_test.py", "r") as f:
    content2 = f.read()
print("Script after code2:", "x = 100" in content2, "y = 200" in content2)


# test_append.py
from pathlib import Path

# Simulate the current logic
script_path = Path("workspace/test_script.py")
code1 = "x = 1"
code2 = "y = 2"

# Current implementation check
if "import" in code2:
    print("Would reinitialize (wrong!)")
else:
    print("Should append")

# Check what actually happens
workspace_setup = "# Setupn"
script_path.write_text(workspace_setup + code1)

current = script_path.read_text()
if code2 not in current:  # This is the check in your code
    script_path.write_text(current + "n" + code2)

print("Final content:", script_path.read_text())

# test_file_location.py
from crew.crew_tools import toolkit, set_session_id
import os
from pathlib import Path

set_session_id("location_test")

# Execute code
code = "x = 999nprint(f'x = {x}')"
result = toolkit.execute_python_code(code)
print("Execution result:", result)

# Check all possible locations
locations = [
    "workspace/default_code_location_test.py",
    "workspace/persistent_code_location_test.py",
    "default_code_location_test.py",
    "persistent_code_location_test.py"
]

print("nSearching for script files:")
for loc in locations:
    if Path(loc).exists():
        print(f"✓ Found: {loc}")
        # Print first 100 chars
        with open(loc, 'r') as f:
            content = f.read()
            print(f"  Content preview: {content[:100]}...")
            print(f"  Contains 'x = 999': {'x = 999' in content}")

# Also check workspace directory
print("nAll files in workspace:")
for f in Path("workspace").glob("*.py"):
    print(f"  - {f.name}")

    # test_sandbox.py
from crew.crew_tools import toolkit, set_session_id

set_session_id("sandbox_test")

# Check if using sandbox
if toolkit.sandbox_manager:
    print("Using SANDBOX execution")
    sandbox = toolkit.sandbox_manager.get_sandbox("sandbox_test")
    
    # Test persistence in sandbox
    result1 = sandbox.execute_python_code("x = 500")
    print("Set x:", result1)
    
    result2 = sandbox.execute_python_code("print(f'x = {x}')")
    print("Read x:", result2)
else:
    print("Using LOCAL execution")




import sys
import os

# Add the project directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crew.crew_tools import (
    ListFilesTool, 
    GetAvailableDatasetsTool, 
    LoadDatasetInfoTool, 
    ExecutePythonCodeTool,
    set_session_id
)

def test_tool_sequence():
    """Test the exact sequence the Data Analyst should follow"""
    
    # Set session ID
    set_session_id("test-session-123")
    
    print("=== Testing Tool Sequence ===")
    
    # 1. List Files
    print("1. Testing List Files Tool:")
    list_tool = ListFilesTool()
    result = list_tool._run()
    print(f"Result: {result}")
    
    # 2. Get Available Datasets
    print("2. Testing Get Available Datasets Tool:")
    datasets_tool = GetAvailableDatasetsTool()
    result = datasets_tool._run()
    print(f"Result: {result}")
    
    # 3. Load Dataset Info
    print("3. Testing Load Dataset Info Tool:")
    info_tool = LoadDatasetInfoTool()
    result = info_tool._run("Housing.csv")  # Use the dataset we know exists
    print(f"Result: {result}")
    
    # 4. Execute Python Code
    print("4. Testing Execute Python Code Tool:")
    code_tool = ExecutePythonCodeTool()
    test_code = """
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Housing.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("First few rows:")
print(df.head())
print("Basic statistics:")
print(df.describe())
"""
    result = code_tool._run(test_code)
    print(f"Result: {result}")

    print("=== All tools tested! ===")

if __name__ == "__main__":
    test_tool_sequence()