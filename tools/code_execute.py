import subprocess
import tempfile
import os
import sys
import re
from typing import Dict, Any

class CodeExecutor:
    """CodeExecutor optimized for real-world data science workflows"""
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Comprehensive allowed imports for data science
        self.allowed_imports = {
            # Standard library
            'math', 'statistics', 'datetime', 'json', 'csv', 'random', 'time',
            'collections', 'itertools', 'functools', 'operator', 'copy', 'pickle',
            're', 'string', 'typing', 'warnings', 'io', 'pathlib',
            
            # Data science core
            'pandas', 'numpy', 'scipy', 'sklearn', 'scikit-learn',
            
            # Machine learning
            'xgboost', 'lightgbm', 'catboost', 'tensorflow', 'torch', 'keras',
            'joblib', 'pickle',
            
            # Visualization
            'matplotlib', 'seaborn', 'plotly', 'bokeh', 'altair',
            
            # Statistical analysis
            'statsmodels', 'pingouin', 'lifelines',
            
            # Text processing
            'nltk', 'spacy', 'gensim', 'transformers', 'textblob',
            
            # Image processing
            'PIL', 'cv2', 'opencv', 'skimage',
            
            # Web and data
            'requests', 'beautifulsoup4', 'bs4', 'lxml', 'html5lib',
            
            # Utilities
            'tqdm', 'logging', 'argparse', 'configparser',
            
            # Jupyter/notebook
            'IPython', 'ipywidgets',
            
            # File handling (controlled)
            'os', 'glob', 'shutil'  # Allow but monitor usage
        }
        
        # Critical security patterns to block
        self.forbidden_patterns = [
            r'import\s+subprocess',      # Prevent system command execution
            r'import\s+sys',             # Prevent Python interpreter manipulation  
            r'__import__\s*\(',          # Prevent dynamic imports
            r'eval\s*\(',                # Prevent code evaluation
            r'exec\s*\(',                # Prevent code execution
            r'compile\s*\(',             # Prevent code compilation
            r'input\s*\(',               # Prevent input blocking
            r'raw_input\s*\(',           # Prevent input blocking
            r'open\s*\(.*/etc/',         # Prevent system file access
            r'open\s*\(.*/proc/',        # Prevent process info access
            r'\.system\s*\(',            # Prevent os.system calls
            r'\.popen\s*\(',             # Prevent process opening
            r'\.spawn\s*\(',             # Prevent process spawning
        ]
        
        # Allowed file operations patterns
        self.safe_file_patterns = [
            r'open\s*\([\'"][^/\'"]+[\'"]',           # Local files only
            r'pd\.read_csv\s*\(',                     # Pandas read operations
            r'np\.load\s*\(',                         # Numpy load operations
            r'\.to_csv\s*\(',                         # Save to CSV
            r'\.to_json\s*\(',                        # Save to JSON
            r'\.save\s*\(',                           # Model save operations
            r'\.dump\s*\(',                           # Pickle dump operations
        ]
    
    def is_safe_code(self, code: str) -> tuple[bool, str]:
        """Enhanced security check for practical data science"""
        
        # Check for strictly forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Security violation: {pattern}"
        
        # Check file operations are safe
        if 'open(' in code:
            # Allow only if it matches safe patterns
            has_safe_file_op = any(
                re.search(pattern, code, re.IGNORECASE) 
                for pattern in self.safe_file_patterns
            )
            
            # Check for dangerous file paths
            dangerous_paths = ['/etc/', '/proc/', '/sys/', '/dev/', '~/', '../']
            has_dangerous_path = any(path in code for path in dangerous_paths)
            
            if has_dangerous_path and not has_safe_file_op:
                return False, "File operation outside workspace not allowed"
        
        # Validate imports (more permissive)
        import_lines = re.findall(r'^\s*(import\s+\S+|from\s+\S+\s+import)', code, re.MULTILINE)
        for imp in import_lines:
            # Extract base module name
            if imp.strip().startswith('import'):
                module = imp.split()[1].split('.')[0]
            else:  # from X import Y
                module = imp.split()[1].split('.')[0]
            
            # Check if module is allowed
            if module not in self.allowed_imports:
                # Special case: allow submodules of allowed packages
                allowed = any(
                    module.startswith(allowed_mod) for allowed_mod in self.allowed_imports
                    if '.' not in allowed_mod  # Only check base modules
                )
                
                if not allowed:
                    return False, f"Import not allowed: {module}"
        
        return True, "Code appears safe"
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code with enhanced workspace support"""
        
        # Safety check
        is_safe, message = self.is_safe_code(code)
        if not is_safe:
            return {
                'success': False,
                'error': f"Security check failed: {message}",
                'output': None
            }
        
        # Enhanced workspace setup
        workspace_setup = f'''
import os
import json
import warnings
warnings.filterwarnings('ignore')  # Suppress common data science warnings

# Workspace setup
WORKSPACE_DIR = "{self.workspace_dir}"
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.chdir(WORKSPACE_DIR)

def save_to_workspace(filename, content):
    """Save content for agent coordination"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, indent=2, ensure_ascii=False)
            else:
                f.write(str(content))
        print(f"Saved to workspace: {{filename}}")
        return filename
    except Exception as e:
        print(f"Error saving {{filename}}: {{e}}")
        return None

def load_from_workspace(filename):
    """Load content from workspace"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
    except FileNotFoundError:
        print(f"File not found: {{filename}}")
        return None
    except Exception as e:
        print(f"Error loading {{filename}}: {{e}}")
        return None

print(f"Working in workspace: {{os.getcwd()}}")

# User code starts here
'''
        
        full_code = workspace_setup + code
        
        # Create temporary file and execute
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            temp_file = f.name
        
        try:
            # Execute with extended timeout for ML operations
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for ML operations
                cwd=self.workspace_dir
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
                'error': "Code execution timed out (5 minutes)",
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

def create_multi_agent_executor(workspace_dir: str = "workspace"):
    """Create executor for multi-agent use"""
    return CodeExecutor(workspace_dir)
