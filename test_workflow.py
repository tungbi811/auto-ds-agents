#!/usr/bin/env python3
"""
Quick test script to verify the workflow and code execution fixes
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

def test_code_execution():
    """Test that code execution works without sandbox"""
    print("🧪 Testing code execution...")
    
    from crew.crew_tools import core_tools, set_session_id
    
    # Set a test session
    set_session_id("test_session_123")
    
    # Test simple code execution
    test_code = """
import pandas as pd
import numpy as np

# Test data manipulation
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)
print("DataFrame created successfully!")
print(f"Shape: {df.shape}")
print(df.head())

# Test calculation
result = df['A'].sum()
print(f"Sum of column A: {result}")
"""
    
    result = core_tools.execute_python_code_raw(test_code)
    
    if result['success']:
        print("✅ Code execution test passed!")
        print(f"Output preview: {result['output'][:200]}...")
        return True
    else:
        print("❌ Code execution test failed!")
        print(f"Error: {result['error']}")
        return False

def test_workflow_basic():
    """Test basic workflow initialization"""
    print("\n🧪 Testing workflow initialization...")
    
    try:
        from workflow.graph import DataScienceWorkflow
        
        # Create workflow
        workflow = DataScienceWorkflow()
        print("✅ Workflow created successfully!")
        
        # Test with basic requirements
        test_requirements = {
            'session_id': 'test_123',
            'business_problem': 'Test housing price analysis',
            'problem_type': 'Regression',
            'success_metrics': ['Model accuracy > 80%'],
            'stakeholders': ['Data scientist', 'Business analyst'],
            'constraints': ['Use only provided dataset'],
            'timeline': '1 week'
        }
        
        print("✅ Workflow initialization test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_housing_dataset():
    """Test if Housing.csv can be loaded"""
    print("\n🧪 Testing Housing.csv dataset loading...")
    
    housing_path = "Housing.csv"
    if not os.path.exists(housing_path):
        print(f"❌ Housing.csv not found at {housing_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(housing_path)
        print(f"✅ Housing dataset loaded successfully!")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Failed to load Housing.csv: {str(e)}")
        return False

def main():
    print("🚀 Running Multi-Agent Workflow Tests\n")
    
    results = []
    
    # Test 1: Code execution
    results.append(test_code_execution())
    
    # Test 2: Workflow initialization  
    results.append(test_workflow_basic())
    
    # Test 3: Dataset loading
    results.append(test_housing_dataset())
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! The workflow should work correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for issues.")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)