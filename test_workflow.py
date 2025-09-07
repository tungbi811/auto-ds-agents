# test_workflow.py - Testing script for LangGraph workflow
import os
import sys
import tempfile
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Add current directory to path for imports
sys.path.append('.')

def create_test_dataset():
    """Create a simple test dataset for validation"""
    data = {
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature_3': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'target': [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        return tmp.name

def test_workflow_initialization():
    """Test that the workflow can be initialized"""
    print("🔧 Testing workflow initialization...")
    
    try:
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Import workflow components
        from agents.workflow import create_codeact_workflow
        from tools.basic_eda import generate_eda_report
        from tools.web_search import web_search
        from tools.document_analyze import analyze_pdf_from_path
        from tools.code_execute import CodeExecutor
        
        tools = {
            'analyze_dataset': generate_eda_report,
            'search_web': web_search,
            'analyze_pdf': analyze_pdf_from_path,
            'execute_code': CodeExecutor().execute_code
        }
        
        workflow = create_codeact_workflow(client, tools)
        print("✅ Workflow initialized successfully")
        return workflow
        
    except Exception as e:
        print(f"❌ Workflow initialization failed: {e}")
        return None

def test_state_management():
    """Test state creation and management"""
    print("\n🔧 Testing state management...")
    
    try:
        from core.state import create_initial_state, DataScienceState
        
        # Create initial state
        state = create_initial_state(
            "Analyze this dataset and build a classification model.",
            dataset_path=create_test_dataset()
        )
        assert isinstance(state, DataScienceState)
        print("✅ State created successfully")
        return state    
    except Exception as e:
        print(f"❌ State management test failed: {e}")
        return None

def test_full_workflow_execution():
    """Test full workflow execution with sample dataset"""
    print("\n🔧 Testing full workflow execution...")
    
    try:
        # Initialize workflow
        workflow = test_workflow_initialization()
        if not workflow:
            return False
            
        # Create test dataset
        dataset_path = create_test_dataset()
        
        # Test workflow execution
        test_prompt = "Analyze this dataset and provide insights about the features and target variable."
        result = workflow.execute(test_prompt, dataset_path)
        
        # Cleanup
        os.unlink(dataset_path)
        
        if result and result.get("success"):
            print("✅ Full workflow execution successful")
            print(f"   Response length: {len(result.get('response', ''))}")
            print(f"   Code operations: {len(result.get('code_history', []))}")
            return True
        else:
            print("❌ Workflow execution failed or incomplete")
            return False
            
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        return False

def test_agent_coordination():
    """Test that agents can coordinate properly"""
    print("\n🔧 Testing agent coordination...")
    
    try:
        from core.state import create_initial_state
        
        # Create state with complex task
        state = create_initial_state(
            "Build a machine learning model to predict the target variable and explain the business implications.",
            dataset_path=create_test_dataset()
        )
        
        # Check that state has proper structure for coordination
        required_fields = ['task', 'current_agent', 'current_phase', 'dataset_path']
        for field in required_fields:
            if not hasattr(state, field):
                print(f"❌ Missing required state field: {field}")
                return False
        
        print("✅ Agent coordination structure validated")
        
        # Cleanup
        if state.dataset_path and os.path.exists(state.dataset_path):
            os.unlink(state.dataset_path)
            
        return True
        
    except Exception as e:
        print(f"❌ Agent coordination test failed: {e}")
        return False

def run_all_tests():
    """Run all workflow tests"""
    print("🧪 Starting workflow tests...\n")
    
    tests = [
        test_workflow_initialization,
        test_state_management,
        test_agent_coordination,
        test_full_workflow_execution
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
    