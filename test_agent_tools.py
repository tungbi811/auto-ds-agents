#!/usr/bin/env python3
"""
Test script to verify agent tools work with dataset detection
"""

import sys
import os
sys.path.append(os.getcwd())

def test_agent_tools():
    """Test that agent tools can detect datasets"""
    print("🧪 Testing agent tools with dataset detection...")
    
    from crew.crew_tools import set_session_id, core_tools
    
    # Set a test session
    set_session_id("test_agent_tools")
    
    print("Testing get_available_datasets...")
    datasets = core_tools.get_available_datasets()
    datasets_result = f"📁 Available datasets in workspace:\n" + "\n".join(f"• {dataset}" for dataset in datasets) if datasets else "❌ No datasets found in workspace."
    print(f"Available datasets result:\n{datasets_result}")
    
    print("\nTesting load_dataset_info with auto-detection...")
    dataset_info = core_tools.load_dataset_info_raw()  # Auto-detect by passing None
    print(f"Dataset info result:\n{dataset_info}")
    
    # Test if the tools work as expected
    if "No datasets found" in datasets_result:
        print("⚠️  No datasets found in workspace")
        return False
    elif "Housing.csv" in datasets_result:
        print("✅ Housing.csv detected successfully!")
        
        if "Failed to load dataset" not in dataset_info:
            print("✅ Dataset info loaded successfully!")
            return True
        else:
            print("❌ Dataset info loading failed")
            return False
    else:
        print("⚠️  Expected Housing.csv not found")
        return False

def main():
    print("🚀 Testing Agent Tools\n")
    
    success = test_agent_tools()
    
    if success:
        print("\n🎉 Agent tools are working correctly!")
        print("Agents should now be able to detect and analyze datasets automatically.")
    else:
        print("\n⚠️  Agent tools test failed.")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)