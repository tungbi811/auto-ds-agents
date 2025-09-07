from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from datetime import datetime
import json
import traceback

class CodeActAgent(ABC):
    """Base class for CodeAct agents that execute Python code"""
    
    def __init__(self, agent_id: str, role: str, llm, tools):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm
        self.tools = tools
    
    @abstractmethod
    def should_execute(self, state) -> bool:
        """Determine if this agent should execute given current state"""
        pass
    
    @abstractmethod
    def get_current_task(self, state) -> str:
        """Get the current task description for this agent"""
        pass
    
    # def execute_action(self, state) -> Dict[str, Any]:
    #     """Main execution method for the agent"""
    #     if not self.should_execute(state):
    #         return state
        
    #     try:
    #         # Get current task
    #         task = self.get_current_task(state)
            
    #         # Generate and execute code
    #         code = self.generate_code(task, state)
    #         result = self.execute_code(code, state)
            
    #         # Process results
    #         if result.get('success', False):
    #             state = self.process_success(state, result)
    #         else:
    #             state = self.process_failure(state, result)
            
    #         return state
            
    #     except Exception as e:
    #         error_msg = f"Agent {self.agent_id} failed: {str(e)}"
    #         state['errors'] = state.get('errors', []) + [error_msg]
    #         return state

    def execute_action(self, state) -> Dict[str, Any]:
        """Main execution method for the agent"""
        print(f"DEBUG: {self.agent_id} received task: {self.get_current_task(state)[:100]}...")

        if not self.should_execute(state):
            return state  
        
        try:
            # Get current task
            task = self.get_current_task(state)
            
            # Generate and execute code
            code = self.generate_code(task, state)
            result = self.execute_code(code, state)
            
            # Process results - ENSURE we always return state
            if result and result.get('success', False):
                updated_state = self.process_success(state, result)
                return updated_state if updated_state is not None else state
            else:
                updated_state = self.process_failure(state, result)
                return updated_state if updated_state is not None else state
            
        except Exception as e:
            error_msg = f"Agent {self.agent_id} failed: {str(e)}"
            state['errors'] = state.get('errors', []) + [error_msg]
            return state

    def generate_code(self, task: str, state) -> str:
        """Generate Python code to accomplish the task"""
        dataset_path = state.get('dataset_path', '')
        
        prompt = f"""
   Generate Python code for this data science task:

    Task: {task}
    Dataset: {dataset_path if dataset_path else "No dataset provided"}

    Requirements:
    1. Use only numerical columns for quantile calculations
    2. Replace deprecated numpy.object with 'object'
    3. Handle mixed data types properly
    4. Include error handling for pandas operations
    5. Focus on basic analysis: shape, dtypes, missing values, basic stats

    Example pattern:
    ```python
    import pandas as pd
    import numpy as np

    # Load data
    df = pd.read_csv("path")

    # Basic info
    print(df.head())
    print(df.info())
    print(df.describe())

    # Only analyze numeric columns for outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print("Numeric analysis:")
        print(df[numeric_cols].describe())
    else:
        print("No numeric columns found")

    Return ONLY the Python code, no explanations or markdown formatting.
"""
        try:
            response = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up code (remove markdown formatting if present)
            if code.startswith('```python'):
                code = code[9:]
            if code.endswith('```'):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            return f"# Error generating code: {str(e)}\nprint('Code generation failed')"
    
    # Generate Python code to accomplish this data science task:

    # Task: {task}

    # Dataset: {dataset_path if dataset_path else "No dataset provided - use sample data"}

    # Requirements:
    # 1. Write complete, executable Python code
    # 2. Import all necessary libraries
    # 3. Handle errors gracefully
    # 4. Print results and save important findings to files

    # Return only the Python code, no explanations.

    def execute_code(self, code: str, state) -> Dict[str, Any]:
        """Execute Python code and return results with proper dict handling"""
        try:
            # Execute code - your tool now returns a dict
            execution_result = self.tools['execute_code'](code)
            
            # Handle dict return format
            if isinstance(execution_result, dict):
                success = execution_result.get('success', False)
                output = execution_result.get('output', '')
                error = execution_result.get('error', '')
            else:
                # Fallback for unexpected format
                success = True
                output = str(execution_result)
                error = ''
            
            # Ensure strings
            output = str(output) if output is not None else ""
            error = str(error) if error is not None else ""
            
            # Record execution in state
            execution_record = {
                'agent': self.agent_id,
                'code': code,
                'output': output,
                'error': error,
                'timestamp': datetime.now().isoformat(),
                'success': success
            }
            
            if 'code_history' not in state:
                state['code_history'] = []
            state['code_history'].append(execution_record)
            
            return {
                'success': success,
                'output': output,
                'error': error,
                'code': code
            }
            
        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            
            # Record failed execution
            execution_record = {
                'agent': self.agent_id,
                'code': code,
                'output': "",
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            
            if 'code_history' not in state:
                state['code_history'] = []
            state['code_history'].append(execution_record)
            
            return {
                'success': False,
                'output': '',
                'error': error_msg,
                'code': code
            }

    @abstractmethod
    def process_success(self, state, result: Dict[str, Any]):
        """Process successful execution results"""
        pass
    
    def process_failure(self, state, result: Dict[str, Any]):
        """Process failed execution results"""
        error_msg = f"{self.agent_id} execution failed: {result.get('error', 'Unknown error')}"
        if 'errors' not in state:
            state['errors'] = []
        state['errors'].append(error_msg)
        return state