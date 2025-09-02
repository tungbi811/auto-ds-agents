import subprocess
import tempfile
import os
import sys
import re
from typing import Dict, Any

class CodeExecutor:
    def __init__(self):
        # Allowed imports for safety
        self.allowed_imports = {
            'math', 'statistics', 'datetime', 'json', 'csv', 'random',
            'pandas', 'numpy', 'matplotlib.pyplot', 'seaborn', 'scipy'
        }
        
        # Forbidden patterns for security
        self.forbidden_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
    
    def is_safe_code(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute"""
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden operation detected: {pattern}"
        
        # Check imports
        import_lines = re.findall(r'^\s*(import\s+\S+|from\s+\S+\s+import)', code, re.MULTILINE)
        for imp in import_lines:
            # Extract module name
            if imp.startswith('import'):
                module = imp.split()[1].split('.')[0]
            else:  # from X import Y
                module = imp.split()[1].split('.')[0]
            
            if module not in self.allowed_imports:
                return False, f"Import not allowed: {module}"
        
        return True, "Code appears safe"
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely and return results"""
        
        # Safety check
        is_safe, message = self.is_safe_code(code)
        if not is_safe:
            return {
                'success': False,
                'error': f"Security check failed: {message}",
                'output': None
            }
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute code with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=tempfile.gettempdir()  # Run in temp directory
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.stderr else None,
                'code': code
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': "Code execution timed out (30s limit)",
                'output': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {str(e)}",
                'output': None
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response (looks for ```python blocks)"""
    
    # Look for ```python code blocks
    python_pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(python_pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    # Look for ``` code blocks
    general_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(general_pattern, text, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
        if any(keyword in code for keyword in ['print(', 'import ', 'def ', 'for ', 'if ']):
            return code
    
    return None

# Add this function to code_execute.py
def handle_code_execution(message, make_ai_request_func, get_prompt_config_func, client):
    """Handle code generation and execution requests - matches other tools"""
    try:
        # 1. Generate code using LLM
        llm_response = make_ai_request_func(message, "code_generation", client)
        
        # 2. Execute the generated code
        code_result = extract_python_code(llm_response)  # Your existing function
        
        if code_result and code_result.get('success'):
            response = f"Here's the solution:\n\n```python\n{code_result.get('code', '')}\n```\n\n**Output:**\n{code_result.get('output', '')}"
        elif code_result:
            response = f"I generated code but there was an error:\n\n**Error:** {code_result.get('error', 'Unknown error')}"
        else:
            response = "Failed to generate or execute code."
            
        return {"response": response, "search_used": False}
        
    except Exception as e:
        return {"response": f"Code execution failed: {str(e)}", "search_used": False}

