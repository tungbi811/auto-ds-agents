import os
import sys
import json
import tempfile
import subprocess
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from ddgs import DDGS

# Try to import Docker functionality
try:
    from tools.sandbox_manager import SandboxManager
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    SandboxManager = None

class DataScienceToolkit:
    """Unified toolkit for multi-agent data science workflows"""
    
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = None
        self.sandbox_manager = None
        self.execution_history = []
    
    def set_session(self, session_id: str):
        """Set session ID and initialize sandbox if available"""
        self.session_id = session_id
        
        if DOCKER_AVAILABLE and SandboxManager:
            try:
                # Initialize SandboxManager with workspace directory only
                self.sandbox_manager = SandboxManager(str(self.workspace_dir))
                print(f"🐳 Sandbox initialized for session: {session_id}")
            except Exception as e:
                print(f"⚠️ Sandbox initialization failed: {e}")
                self.sandbox_manager = None
    
    # ==================== DATASET MANAGEMENT ====================
    
    def get_available_datasets(self) -> List[str]:
        """Get list of dataset files in workspace"""
        extensions = ['*.csv', '*.xlsx', '*.parquet']
        datasets = []
        for ext in extensions:
            datasets.extend([f.name for f in self.workspace_dir.glob(ext)])
        return sorted(datasets)

    def load_dataset_info(self, dataset_name: str = None) -> str:
        """Load and analyze dataset structure"""
        if not dataset_name:
            datasets = self.get_available_datasets()
            if not datasets:
                return "❌ No datasets found in workspace"
            dataset_name = datasets[0]  # Use first available
        
        file_path = self.workspace_dir / dataset_name
        if not file_path.exists():
            return f"❌ Dataset '{dataset_name}' not found"
        
        try:
            # Load based on extension
            if dataset_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif dataset_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif dataset_name.endswith('.json'):
                df = pd.read_json(file_path)
            elif dataset_name.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                return f"❌ Unsupported file format: {dataset_name}"
            
            # Generate info
            info = f"""📊 Dataset: {dataset_name}
📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns
📋 Columns: {', '.join(df.columns.tolist())}
🔢 Data types: {df.dtypes.value_counts().to_dict()}
❓ Missing values: {df.isnull().sum().sum()}
🔄 Duplicates: {df.duplicated().sum()}

📈 Sample data:
{df.head(3).to_string()}"""
            
            return info
            
        except Exception as e:
            return f"❌ Error loading dataset: {str(e)}"
    
    # ==================== FILE OPERATIONS ====================
    
    def write_file(self, filename: str, content: str) -> str:
        """Write content to workspace file"""
        try:
            file_path = self.workspace_dir / Path(filename).name
            file_path.write_text(content, encoding='utf-8')
            return f"✅ Wrote {len(content)} characters to {filename}"
        except Exception as e:
            return f"❌ Error writing {filename}: {str(e)}"
    
    def read_file(self, filename: str) -> str:
        """Read content from workspace file"""
        try:
            file_path = self.workspace_dir / Path(filename).name
            if not file_path.exists():
                return f"❌ File '{filename}' not found"
            return file_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"❌ Error reading {filename}: {str(e)}"
    
    def list_files(self) -> str:
        """List all files in workspace"""
        try:
            files = [f.name for f in self.workspace_dir.iterdir() if f.is_file()]
            if not files:
                return "📂 Workspace is empty"
            return "📁 Workspace files:\n" + "\n".join(f"• {f}" for f in sorted(files))
        except Exception as e:
            return f"❌ Error listing files: {str(e)}"
    
    # ==================== WEB SEARCH ====================
    
    def search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web for information"""
        try:
            results = DDGS().text(query, max_results=max_results)
            if not results:
                return f"❌ No results found for: {query}"
            
            formatted = f"🔍 Search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                url = result.get('href', 'No URL')
                body = result.get('body', '')[:150]
                formatted += f"{i}. {title}\n   {url}\n   {body}...\n\n"
            
            return formatted
        except Exception as e:
            return f"❌ Web search failed: {str(e)}"
    
    # ==================== CODE EXECUTION ====================
    
    def execute_python_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code with sandbox fallback"""
        # Try sandbox first
        if self.session_id and self.sandbox_manager:
            try:
                sandbox = self.sandbox_manager.get_sandbox(self.session_id)
                result = sandbox.execute_python_code(code)
                if result.get('success'):
                    self._log_execution('sandbox', code, result)
                    return result
            except Exception as e:
                print(f"⚠️ Sandbox failed, using local: {e}")
        
        # Local execution fallback
        return self._execute_local_python(code)

    def _execute_local_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code locally with workspace setup"""
        if not self._is_safe_code(code):
            return {
                'success': False,
                'output': '',
                'error': 'Code blocked for security reasons'
            }
        
        # Workspace setup with dynamic dataset loading
        workspace_setup = f'''
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import json

# Data science imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler  
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE_DIR = "{str(self.workspace_dir.absolute())}"
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.chdir(WORKSPACE_DIR)
print(f"Working directory: {{os.getcwd()}}")
print(f"Available files: {{os.listdir('.')}}")

'''
        
        full_code = workspace_setup + code
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(full_code)
                temp_file = f.name
            
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.workspace_dir.absolute())
            )
            
            execution_result = {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.stderr else '',
                'exit_code': result.returncode
            }
            
            self._log_execution('local', code, execution_result)
            return execution_result
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Code execution timed out (5 minutes)'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Execution error: {str(e)}'
            }
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def execute_shell_command(self, command: str) -> Dict[str, Any]:
        """Execute shell command with security checks"""
        if not self._is_safe_command(command):
            return {
                'success': False,
                'output': '',
                'error': f'Command blocked for security: {command}'
            }
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_dir)
            )
            
            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.stderr else ''
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': '',
                'error': 'Command timed out (30 seconds)'
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f'Command execution error: {str(e)}'
            }
    
    # ==================== SECURITY & UTILITIES ====================
    
    def _is_safe_code(self, code: str) -> bool:
        """Check if Python code is safe to execute"""
        dangerous_patterns = [
            'import subprocess', 'os.system', 'eval(', 'exec(',
            '__import__', 'open(', 'file(', 'input(', 'raw_input(',
            'sys.exit', 'quit()', 'exit()'
        ]
        code_lower = code.lower()
        return not any(pattern.lower() in code_lower for pattern in dangerous_patterns)
    
    def _is_safe_command(self, command: str) -> bool:
        """Check if shell command is safe to execute"""
        dangerous_patterns = [
            'rm -rf', 'format', 'fdisk', 'mkfs', 'dd if=',
            'passwd', 'sudo', 'su -', 'chmod 777'
        ]
        command_lower = command.lower()
        return not any(pattern in command_lower for pattern in dangerous_patterns)
    
    def _log_execution(self, method: str, code: str, result: Dict[str, Any]):
        """Log code execution for debugging"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'success': result.get('success', False),
            'code': code[:100] + '...' if len(code) > 100 else code,
            'output': result.get('output', '')[:200],
            'error': result.get('error', '')
        }
        self.execution_history.append(log_entry)
    
    def cleanup_session(self):
        """Clean up session resources"""
        if self.sandbox_manager:
            try:
                self.sandbox_manager.cleanup_session(self.session_id)
                print(f"🧹 Session {self.session_id} cleaned up")
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")

# ==================== GLOBAL INSTANCE & TOOLS ====================

toolkit = DataScienceToolkit()

def set_session_id(session_id: str):
    """Set session ID for the toolkit"""
    print(f"🔧 Setting global session ID: {session_id}")
    toolkit.set_session(session_id)

class WriteFileTool(BaseTool):
    name: str = "Write File"
    description: str = "Write content to a file in the workspace"
    
    def _run(self, filename: str, content: str) -> str:
        return toolkit.write_file(filename, content)

class ReadFileTool(BaseTool):
    name: str = "Read File"
    description: str = "Read content from a file in the workspace"
    
    def _run(self, filename: str) -> str:
        return toolkit.read_file(filename)

class ListFilesTool(BaseTool):
    name: str = "List Files"
    description: str = "List all files in the workspace"

    def _run(self) -> str:
        return toolkit.list_files()

class SearchWebTool(BaseTool):
    name: str = "Search Web"
    description: str = "Search the web for information"

    def _run(self, query: str) -> str:
        return toolkit.search_web(query)

class ExecutePythonCodeTool(BaseTool):
    name: str = "Execute Python Code"
    description: str = "Execute Python code in a sandboxed environment"

    def _run(self, code: str) -> str:
        try:
            result = toolkit.execute_python_code(code)

            if result.get('success'):
                output = result.get('output', '')
                return f"✅ Code executed successfully.\n\nOutput:\n{output}"
            else:
                error = result.get('error', 'Unknown error')
                return f"❌ Code execution failed.\n\nError:\n{error}"
            
        except Exception as e:
            return f"❌ Execution error: {str(e)}"

class ExecuteShellCommandTool(BaseTool):
    name: str = "Execute Shell Command"
    description: str = "Execute a shell command in the workspace"   

    def _run(self, command: str) -> str:
        try:
            result = toolkit.execute_shell_command(command)
            if result.get('success'):
                output = result.get('output', '')
                return f"✅ Command executed: {command}\n\nOutput:\n{output}"
            else:
                error = result.get('error', 'Unknown error')
                return f"❌ Command failed: {command}\n\nError:\n{error}"
            
        except Exception as e:
            return f"❌ Command error: {str(e)}"

class LoadDatasetInfoTool(BaseTool):
    name: str = "Load Dataset Info"
    description: str = "Load and analyze dataset structure and information"
    
    def _run(self, dataset_name: str = None) -> str:
        try:
            # If no specific dataset provided, find uploaded datasets
            if not dataset_name:
                available = toolkit.get_available_datasets()
                if not available:
                    return "❌ No datasets found in workspace. Please upload a dataset first."
                
                # Use the first CSV file if available, or first file otherwise
                csv_files = [f for f in available if f.endswith('.csv')]
                dataset_name = csv_files[0] if csv_files else available[0]

            return toolkit.load_dataset_info(dataset_name)
        except Exception as e:
            return f"❌ Error analyzing dataset: {str(e)}"

class GetAvailableDatasetsTool(BaseTool):
    name: str = "Get Available Datasets"
    description: str = "Get list of available datasets in the workspace"

    def _run(self) -> str:
        datasets = toolkit.get_available_datasets()
        if not datasets:
            return "❌ No datasets found in workspace"
        return "📁 Available datasets:\n" + "\n".join(f"• {dataset}" for dataset in datasets)

def get_tools_for_agent(agent_type: str) -> List:
    """Get appropriate tools for different agent types"""
    tool_sets = {
        "business_analyst": [WriteFileTool(), ReadFileTool(), ListFilesTool(), SearchWebTool()],
        "project_manager": [WriteFileTool(), ReadFileTool(), ListFilesTool(), GetAvailableDatasetsTool(), LoadDatasetInfoTool()],
        "data_analyst": [ExecutePythonCodeTool(), ExecuteShellCommandTool(), LoadDatasetInfoTool(), GetAvailableDatasetsTool(), WriteFileTool(), ReadFileTool(), ListFilesTool()],
        "ml_engineer": [ExecutePythonCodeTool(), ExecuteShellCommandTool(), LoadDatasetInfoTool(), GetAvailableDatasetsTool(), WriteFileTool(), ReadFileTool(), ListFilesTool()],
        "business_translator": [ReadFileTool(), WriteFileTool(), ListFilesTool(), SearchWebTool()]
    }

    return tool_sets.get(agent_type, [WriteFileTool(), ReadFileTool(), ListFilesTool()])

def cleanup_session():
    """Clean up current session"""
    toolkit.cleanup_session()

__all__ = [
    'ExecutePythonCodeTool', 'ExecuteShellCommandTool', 'SearchWebTool',
    'LoadDatasetInfoTool', 'GetAvailableDatasetsTool',
    'WriteFileTool', 'ReadFileTool', 'ListFilesTool',
    'get_tools_for_agent', 'set_session_id', 'cleanup_session'
]