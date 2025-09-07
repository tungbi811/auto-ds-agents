# core/memory.py
import os
import json
import shutil
from typing import Dict, Any, List, Optional
from datetime import datetime

class WorkspaceManager:
    """Manages file-based memory workspace for agents (inspired by Manus approach)"""
    
    def __init__(self, base_dir: str = "workspace"):
        self.base_dir = base_dir
        self.ensure_workspace_exists()
    
    def ensure_workspace_exists(self):
        """Ensure workspace directory exists"""
        os.makedirs(self.base_dir, exist_ok=True)
    
    def get_session_workspace(self, session_id: str) -> str:
        """Get workspace directory for specific session"""
        session_dir = os.path.join(self.base_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    def save_file(self, session_id: str, filename: str, content: Any) -> str:
        """Save content to workspace file"""
        workspace_dir = self.get_session_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                if isinstance(content, (dict, list)):
                    json.dump(content, f, indent=2, ensure_ascii=False)
                else:
                    f.write(str(content))
            
            return filepath
        except Exception as e:
            raise Exception(f"Failed to save file {filename}: {str(e)}")
    
    def load_file(self, session_id: str, filename: str) -> Optional[Any]:
        """Load content from workspace file"""
        workspace_dir = self.get_session_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
        except Exception as e:
            raise Exception(f"Failed to load file {filename}: {str(e)}")
    
    def list_files(self, session_id: str) -> List[str]:
        """List all files in session workspace"""
        workspace_dir = self.get_session_workspace(session_id)
        
        try:
            return [f for f in os.listdir(workspace_dir) 
                   if os.path.isfile(os.path.join(workspace_dir, f))]
        except FileNotFoundError:
            return []
    
    def delete_file(self, session_id: str, filename: str) -> bool:
        """Delete file from workspace"""
        workspace_dir = self.get_session_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception:
            return False
    
    def cleanup_session(self, session_id: str):
        """Clean up entire session workspace"""
        workspace_dir = self.get_session_workspace(session_id)
        
        try:
            if os.path.exists(workspace_dir):
                shutil.rmtree(workspace_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup session {session_id}: {str(e)}")
    
    def get_file_metadata(self, session_id: str, filename: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a workspace file"""
        workspace_dir = self.get_session_workspace(session_id)
        filepath = os.path.join(workspace_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            stat = os.stat(filepath)
            return {
                'filename': filename,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'path': filepath
            }
        except Exception:
            return None

class TodoManager:
    """Manages todo.md file for tracking progress (inspired by Manus)"""
    
    def __init__(self, workspace_manager: WorkspaceManager):
        self.workspace = workspace_manager
        self.todo_filename = "todo.md"
    
    def create_todo(self, session_id: str, user_request: str, phases: List[str]) -> str:
        """Create initial todo.md file"""
        content = f"""# Data Science Project Todo

**Request:** {user_request}
**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Phases

"""
        
        for i, phase in enumerate(phases, 1):
            content += f"{i}. [ ] {phase}\n"
        
        content += f"""
## Progress Log

- Project started at {datetime.now().strftime('%H:%M:%S')}

"""
        
        self.workspace.save_file(session_id, self.todo_filename, content)
        return content
    
    def update_todo(self, session_id: str, phase: str, status: str = "completed", notes: str = "") -> str:
        """Update todo.md with progress"""
        current_content = self.workspace.load_file(session_id, self.todo_filename) or ""
        
        # Update the specific phase
        lines = current_content.split('\n')
        updated_lines = []
        
        for line in lines:
            if phase.lower() in line.lower() and "[ ]" in line:
                # Mark as completed
                updated_line = line.replace("[ ]", "[x]")
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        
        # Add progress log entry
        timestamp = datetime.now().strftime('%H:%M:%S')
        progress_entry = f"- {timestamp}: {phase} {status}"
        if notes:
            progress_entry += f" - {notes}"
        
        # Insert before the last empty line
        updated_lines.insert(-1, progress_entry)
        
        updated_content = '\n'.join(updated_lines)
        self.workspace.save_file(session_id, self.todo_filename, updated_content)
        
        return updated_content
    
    def get_progress_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of progress from todo.md"""
        content = self.workspace.load_file(session_id, self.todo_filename)
        
        if not content:
            return {"completed": 0, "total": 0, "phases": []}
        
        lines = content.split('\n')
        total_phases = 0
        completed_phases = 0
        phases = []
        
        for line in lines:
            if ". [" in line:  # Phase line
                total_phases += 1
                is_completed = "[x]" in line
                if is_completed:
                    completed_phases += 1
                
                # Extract phase name
                phase_name = line.split("] ")[-1] if "] " in line else line
                phases.append({
                    "name": phase_name,
                    "completed": is_completed
                })
        
        return {
            "completed": completed_phases,
            "total": total_phases,
            "completion_rate": completed_phases / total_phases if total_phases > 0 else 0,
            "phases": phases
        }