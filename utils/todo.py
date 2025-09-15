from crew.crew_tools import toolkit
from datetime import datetime
from typing import List, Dict, Any

class TodoManager:
    """Single todo manager using direct file operations like Manus"""
    
    @staticmethod
    def create_initial_todo(user_request: str, project_plan: Dict[str, Any]) -> str:
        """Create initial todo.md based on project plan - Manus style"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        task_assignments = project_plan.get('task_assignments', [])
        
        # Create todo content with specific, actionable steps
        todo_content = f"""# Project Todo List

**Request:** {user_request}
**Created:** {timestamp}
**Status:** In Progress

## Task Breakdown

"""
        
        # Convert task assignments to actionable todo items
        for i, task in enumerate(task_assignments, 1):
            agent_type = task.get('agent_type', 'Unknown')
            description = task.get('task_description', 'No description')
            priority = task.get('priority', 3)
            
            todo_content += f"""### {i}. {agent_type} - Priority {priority}/5
- [ ] {description}
- [ ] Generate deliverables and save results
- [ ] Update progress in todo.md

"""
        
        # Add completion tracking
        todo_content += f"""
## Completion Status

**Tasks Remaining:** {len(task_assignments)}
**Tasks Completed:** 0
**Overall Progress:** 0%

## Execution Log

- {datetime.now().strftime('%H:%M:%S')}: Project initiated
"""
        
        # Save using core_tools file operations
        result = toolkit.write_file("todo.md", todo_content)
        return f"Initial todo.md created with {len(task_assignments)} tasks"
    
    @staticmethod
    def update_task_progress(task_description: str, status: str, agent_name: str = "", notes: str = "") -> str:
        """Update specific task progress in todo.md - Manus style"""
        
        # Read current todo.md
        current_todo = toolkit.read_file("todo.md")
        
        if "not found" in current_todo:
            return "Error: todo.md file not found"
        
        # Extract content (remove the "Content of todo.md:" prefix)
        if "Content of todo.md:" in current_todo:
            todo_lines = current_todo.split("Content of todo.md:")[1].strip().split('\n')
        else:
            todo_lines = current_todo.split('\n')
        
        updated_lines = []
        task_found = False
        
        # Update the specific task
        for line in todo_lines:
            if task_description.lower() in line.lower() and "- [ ]" in line:
                # Mark as completed
                updated_line = line.replace("- [ ]", "- [x]")
                updated_lines.append(updated_line)
                task_found = True
            else:
                updated_lines.append(line)
        
        if not task_found:
            # Add new task if not found
            updated_lines.append(f"- [x] {task_description} ({status})")
        
        # Add execution log entry
        timestamp = datetime.now().strftime('%H:%M:%S')
        agent_info = f" by {agent_name}" if agent_name else ""
        log_entry = f"- {timestamp}: {task_description} {status}{agent_info}"
        if notes:
            log_entry += f" - {notes}"
        
        # Find execution log section and add entry
        for i, line in enumerate(updated_lines):
            if "## Execution Log" in line:
                updated_lines.insert(i + 1, log_entry)
                break
        else:
            # Add execution log if not found
            updated_lines.extend([
                "",
                "## Execution Log",
                log_entry
            ])
        
        # Update completion statistics
        total_tasks = sum(1 for line in updated_lines if "- [" in line and ("[ ]" in line or "[x]" in line))
        completed_tasks = sum(1 for line in updated_lines if "- [x]" in line)
        remaining_tasks = total_tasks - completed_tasks
        progress = int((completed_tasks / total_tasks * 100)) if total_tasks > 0 else 0
        
        # Update status section
        updated_content = []
        for line in updated_lines:
            if "**Tasks Remaining:**" in line:
                updated_content.append(f"**Tasks Remaining:** {remaining_tasks}")
            elif "**Tasks Completed:**" in line:
                updated_content.append(f"**Tasks Completed:** {completed_tasks}")
            elif "**Overall Progress:**" in line:
                updated_content.append(f"**Overall Progress:** {progress}%")
            else:
                updated_content.append(line)
        
        # Save updated todo
        updated_todo = '\n'.join(updated_content)
        toolkit.write_file("todo.md", updated_todo)

        return f"Task '{task_description}' marked as {status}"
    
    @staticmethod
    def mark_phase_complete(phase_name: str, agent_name: str) -> str:
        """Mark an entire phase as complete - Manus style"""
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Add completion entry to log
        completion_note = f"Phase '{phase_name}' completed by {agent_name}"
        TodoManager.update_task_progress(
            task_description=phase_name,
            status="COMPLETED",
            agent_name=agent_name,
            notes="Phase finished successfully"
        )
        
        return f"Phase '{phase_name}' marked complete"
    
    @staticmethod
    def get_todo_summary() -> Dict[str, Any]:
        """Get current todo status summary"""

        current_todo = toolkit.read_file("todo.md")

        if "not found" in current_todo:
            return {"error": "No todo.md file found"}
        
        # Extract content
        if "Content of todo.md:" in current_todo:
            content = current_todo.split("Content of todo.md:")[1].strip()
        else:
            content = current_todo
        
        lines = content.split('\n')
        
        # Count tasks
        total_tasks = sum(1 for line in lines if "- [" in line and ("[ ]" in line or "[x]" in line))
        completed_tasks = sum(1 for line in lines if "- [x]" in line)
        progress = int((completed_tasks / total_tasks * 100)) if total_tasks > 0 else 0
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "remaining_tasks": total_tasks - completed_tasks,
            "progress_percentage": progress,
            "is_complete": progress == 100
        }
    
    @staticmethod
    def add_execution_note(note: str, agent_name: str = "") -> str:
        """Add a note to the execution log - Manus style"""
        
        current_todo = toolkit.read_file("todo.md")
        
        if "not found" in current_todo:
            return "Error: todo.md file not found"
        
        # Extract content
        if "Content of todo.md:" in current_todo:
            content = current_todo.split("Content of todo.md:")[1].strip()
        else:
            content = current_todo
        
        # Add timestamped note
        timestamp = datetime.now().strftime('%H:%M:%S')
        agent_info = f" ({agent_name})" if agent_name else ""
        log_entry = f"- {timestamp}: {note}{agent_info}"
        
        # Find execution log and add entry
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "## Execution Log" in line:
                lines.insert(i + 1, log_entry)
                break
        else:
            lines.extend(["", "## Execution Log", log_entry])
        
        updated_content = '\n'.join(lines)
        toolkit.write_file("todo.md", updated_content)

        return "Note added to execution log"

# Static access pattern like Manus
todo_manager = TodoManager()