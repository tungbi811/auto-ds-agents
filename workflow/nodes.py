import json
import os
from pathlib import Path
from datetime import datetime
from crew.agents.crew_agents import MultiAgent
from crew.crew_tools import set_session_id, toolkit
from utils.todo import TodoManager
from langchain_core.messages import AIMessage
from typing import Dict, Any

class DataScienceNodes:
    """Node implementations for LangGraph workflow"""
    
    def __init__(self):
        """Initialize with workspace setup"""
        os.makedirs("workspace", exist_ok=True)
        self.crew_workflow = None
        self.todo_manager = TodoManager()
    
    def load_requirements(self, state):
        """Load requirements from chatbot JSON file"""
        print("# Loading requirements from chatbot")
        
        if state.get('requirements'):
            print(f"## Using existing requirements for session: {state.get('session_id', 'unknown')}")
            return {
                **state,
                "workflow_status": "requirements_loaded",
                "current_agent": "business_analyst"
            }
        
        workspace_dir = Path("workspace")
        if not workspace_dir.exists():
            return {
                **state,
                "workflow_status": "failed",
                "messages": [AIMessage(content="❌ No workspace directory found")]
            }
        
        req_files = list(workspace_dir.glob("requirements_*.json"))
        if not req_files:
            return {
                **state,
                "workflow_status": "failed", 
                "messages": [AIMessage(content="❌ No requirements file found. Please complete information gathering first.")]
            }
        
        latest_req_file = max(req_files, key=os.path.getctime)
        
        try:
            with open(latest_req_file, 'r') as f:
                req_data = json.load(f)
            
            session_id = req_data.get('session_id', 'default')
            set_session_id(session_id)
            
            print(f"## Loaded requirements from {latest_req_file.name}")
            
            return {
                **state,
                "requirements": req_data.get('requirements', {}),
                "session_id": session_id,
                "workflow_status": "requirements_loaded",
                "current_agent": "business_analyst",
                "messages": [AIMessage(content=f"✅ Requirements loaded from {latest_req_file.name}")]
            }
            
        except Exception as e:
            print(f"## Error loading requirements: {str(e)}")
            return {
                **state,
                "workflow_status": "failed",
                "messages": [AIMessage(content=f"❌ Error loading requirements: {str(e)}")]
            }
    
    def business_analyst_node(self, state):
        """Execute Business Analyst using CrewAI agent"""
        print("# Executing Business Analyst")
        
        try:
            if not self.crew_workflow:
                session_id = state.get('session_id', 'default_session')
                self.crew_workflow = MultiAgent(session_id=session_id)
            
            business_analyst = self.crew_workflow.business_analyst()
            requirements = state.get('requirements', {})
            
            context = f"""
            Business Requirements Context:
            - Problem: {requirements.get('business_problem', 'Not specified')}
            - Success Metrics: {', '.join(requirements.get('success_metrics', []))}
            - Timeline: {requirements.get('timeline', 'Not specified')}
            - Stakeholders: {', '.join(requirements.get('stakeholders', []))}
            - Problem Type: {requirements.get('problem_type', 'Not specified')}
            - Constraints: {', '.join(requirements.get('constraints', []))}
            """
            
            task = self.crew_workflow.understand_task()
            result = business_analyst.execute_task(task, context=context)
            output = toolkit.write_file("/workspace/business_issue.md", result)

            if not (Path("workspace") / "business_issue.md").exists():
                raise Exception("Failed to create business_issue.md")
            
            try:
                TodoManager.mark_phase_complete("business_analyst", "Business Analyst")
                summary = TodoManager.get_todo_summary()
                completed_count = summary.get("completed_tasks", 1)
                total_count = summary.get("total_tasks", 5)
                progress = summary.get("progress_percentage", 20)
            except Exception as e:
                print(f"## Todo operation failed: {str(e)}")
                completed_count = 1
                total_count = 5
                progress = 20
            
            return {
                **state,
                "business_analysis": {
                    "result": str(result),
                    "completed_at": datetime.now().isoformat()
                },
                "completed_tasks": state.get('completed_tasks', []) + ['business_analyst'],
                "current_agent": "project_manager",
                "workflow_status": "business_analysis_complete",
                "workspace_files": state.get('workspace_files', []) + ['business_issue.md'],
                "messages": state.get('messages', []) + [AIMessage(content="✅ Business analysis completed")],
                "completed_tasks_count": completed_count,
                "total_tasks_count": total_count,
                "project_progress": progress
            }
            
        except Exception as e:
            print(f"## Business Analyst error: {str(e)}")
            return {
                **state,
                "workflow_status": "failed",
                "messages": state.get('messages', []) + [AIMessage(content=f"❌ Business Analyst failed: {str(e)}")]
            }
    
    def project_manager_node(self, state):
        """Execute Project Manager using CrewAI agent"""
        print("# Executing Project Manager")
        
        try:
            if not self.crew_workflow:
                session_id = state.get('session_id', 'default_session')
                self.crew_workflow = MultiAgent(session_id=session_id)
            
            project_manager = self.crew_workflow.project_manager()
            task = self.crew_workflow.plan_task()
            
            result = project_manager.execute_task(task)
            
            try:
                user_request = state.get("requirements", {}).get("business_problem", "Multi-agent data science workflow")
                project_plan = {"task_assignments": [
                    {"agent_type": "Business Analyst", "task_description": "Analyze business requirements", "priority": 5},
                    {"agent_type": "Project Manager", "task_description": "Create project plan", "priority": 5},
                    {"agent_type": "Data Analyst", "task_description": "Perform data analysis", "priority": 4},
                    {"agent_type": "ML Engineer", "task_description": "Build ML models", "priority": 4},
                    {"agent_type": "Business Translator", "task_description": "Generate business recommendations", "priority": 3}
                ]}
                TodoManager.create_initial_todo(user_request, project_plan)
                summary = TodoManager.get_todo_summary()
                completed_count = summary.get("completed_tasks", 2)
                total_count = summary.get("total_tasks", 5)
                progress = summary.get("progress_percentage", 40)
            except Exception as e:
                print(f"## Todo operation failed: {str(e)}")
                completed_count = 2
                total_count = 5
                progress = 40

            return {
                **state,
                "project_plan": {
                    "result": str(result),
                    "completed_at": datetime.now().isoformat()
                },
                "completed_tasks": state.get('completed_tasks', []) + ['project_manager'],
                "current_agent": "data_analyst",
                "workflow_status": "project_planning_complete",
                "workspace_files": state.get('workspace_files', []) + ['todo.md'],
                "messages": state.get('messages', []) + [AIMessage(content="✅ Project planning completed")],
                "completed_tasks_count": completed_count,
                "total_tasks_count": total_count,
                "project_progress": progress
            }
            
        except Exception as e:
            print(f"## Project Manager error: {str(e)}")
            return {
                **state,
                "workflow_status": "failed", 
                "messages": state.get('messages', []) + [AIMessage(content=f"❌ Project Manager failed: {str(e)}")]
            }
    
    def data_analyst_node(self, state):
        """Execute Data Analyst using CrewAI agent"""
        print("# Executing Data Analyst")
        
        try:
            if not self.crew_workflow:
                session_id = state.get('session_id', 'default_session')
                self.crew_workflow = MultiAgent(session_id=session_id)

            data_analyst = self.crew_workflow.data_analyst()
            task = self.crew_workflow.data_task()
            
            state["dataset_path"] = toolkit.get_available_datasets()[0] if toolkit.get_available_datasets() else None
            
            result = data_analyst.execute_task(task)
            
            # CRITICAL: Verify both files were created
            data_report_path = Path("workspace") / "data_analysis_report.md"
            cleaned_data_path = Path("workspace") / "cleaned_dataset.csv"
            
            missing_files = []
            if not data_report_path.exists():
                missing_files.append("data_analysis_report.md")
            if not cleaned_data_path.exists():
                missing_files.append("cleaned_dataset.csv")
                
            if missing_files:
                print(f"❌ CRITICAL ERROR: Missing files: {', '.join(missing_files)}")
                print(f"❌ Agent result was: {str(result)[:200]}...")
                # Show what files were actually created
                workspace_files = list(Path("workspace").glob("*"))
                print(f"❌ Files in workspace: {[f.name for f in workspace_files]}")
                raise Exception(f"Data Analyst failed to create required files: {missing_files}")
            else:
                print("✅ data_analysis_report.md successfully created")
                print("✅ cleaned_dataset.csv successfully created")
                
                # Check file content sizes
                report_content = data_report_path.read_text()
                print(f"ℹ️ Report file size: {len(report_content)} characters")
                
                # Check cleaned dataset size
                try:
                    import pandas as pd
                    cleaned_df = pd.read_csv(cleaned_data_path)
                    print(f"ℹ️ Cleaned dataset shape: {cleaned_df.shape}")
                    print("✅ Both deliverables created successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not validate cleaned dataset: {e}")
                    
            try:
                TodoManager.mark_phase_complete("data_analyst", "Data Analyst")
                summary = TodoManager.get_todo_summary()
                completed_count = summary.get("completed_tasks", 3)
                total_count = summary.get("total_tasks", 5)
                progress = summary.get("progress_percentage", 60)
            except Exception as e:
                print(f"## Todo operation failed: {str(e)}")
                completed_count = 3
                total_count = 5
                progress = 60

            return {
                **state,
                "data_analysis": {
                    "result": str(result),
                    "completed_at": datetime.now().isoformat()
                },
                "completed_tasks": state.get('completed_tasks', []) + ['data_analyst'],
                "current_agent": "ml_engineer",
                "workflow_status": "data_analysis_complete",
                "workspace_files": state.get('workspace_files', []) + ['data_analysis_report.md'],
                "messages": state.get('messages', []) + [AIMessage(content="✅ Data analysis completed")],
                "completed_tasks_count": completed_count,
                "total_tasks_count": total_count,
                "project_progress": progress
            }
            
        except Exception as e:
            print(f"## Data Analyst error: {str(e)}")
            return {
                **state,
                "workflow_status": "failed",
                "messages": state.get('messages', []) + [AIMessage(content=f"❌ Data Analyst failed: {str(e)}")]
            }
    
    def ml_engineer_node(self, state):
        """Execute ML Engineer using CrewAI agent"""
        print("# Executing ML Engineer")
        
        try:
            if not self.crew_workflow:
                session_id = state.get('session_id', 'default_session')
                self.crew_workflow = MultiAgent(session_id=session_id)

            ml_engineer = self.crew_workflow.ml_engineer()
            task = self.crew_workflow.model_task()
            
            result = ml_engineer.execute_task(task)
            
            # CRITICAL: Verify that model_report.md was actually created
            model_report_path = Path("workspace") / "model_report.md"
            if not model_report_path.exists():
                print("❌ CRITICAL ERROR: model_report.md was not created!")
                print("❌ ML Engineer failed to execute Python code properly")
                raise Exception("ML Engineer failed to create model_report.md - likely didn't execute Python code or build models")
            else:
                print("✅ model_report.md successfully created")
                # Check if the file has actual content
                report_content = model_report_path.read_text()
                if len(report_content) < 500:  # Basic sanity check
                    print("⚠️ Warning: model_report.md seems very short - may not contain real model results")
            
            try:
                TodoManager.mark_phase_complete("ml_engineer", "ML Engineer")
                summary = TodoManager.get_todo_summary()
                completed_count = summary.get("completed_tasks", 4)
                total_count = summary.get("total_tasks", 5)
                progress = summary.get("progress_percentage", 80)
            except Exception as e:
                print(f"## Todo operation failed: {str(e)}")
                completed_count = 4
                total_count = 5
                progress = 80

            return {
                **state,
                "model_results": {
                    "result": str(result),
                    "completed_at": datetime.now().isoformat()
                },
                "completed_tasks": state.get('completed_tasks', []) + ['ml_engineer'],
                "current_agent": "business_translator",
                "workflow_status": "model_development_complete",
                "workspace_files": state.get('workspace_files', []) + ['model_report.md'],
                "messages": state.get('messages', []) + [AIMessage(content="✅ Model development completed")],
                "completed_tasks_count": completed_count,
                "total_tasks_count": total_count,
                "project_progress": progress
            }
            
        except Exception as e:
            print(f"## ML Engineer error: {str(e)}")
            return {
                **state,
                "workflow_status": "failed",
                "messages": state.get('messages', []) + [AIMessage(content=f"❌ ML Engineer failed: {str(e)}")]
            }
    
    def business_translator_node(self, state):
        """Execute Business Translator using CrewAI agent"""
        print("# Executing Business Translator")
        
        try:
            if not self.crew_workflow:
                session_id = state.get('session_id', 'default_session')
                self.crew_workflow = MultiAgent(session_id=session_id)

            business_translator = self.crew_workflow.business_translator()
            task = self.crew_workflow.translate_task()
            
            result = business_translator.execute_task(task)
            
            try:
                TodoManager.mark_phase_complete("business_translator", "Business Translator")
                summary = TodoManager.get_todo_summary()
                completed_count = summary.get("completed_tasks", 5)
                total_count = summary.get("total_tasks", 5)
                progress = summary.get("progress_percentage", 100)
            except Exception as e:
                print(f"## Todo operation failed: {str(e)}")
                completed_count = 5
                total_count = 5
                progress = 100
            
            return {
                **state,
                "business_recommendations": {
                    "result": str(result),
                    "completed_at": datetime.now().isoformat()
                },
                "completed_tasks": state.get('completed_tasks', []) + ['business_translator'],
                "current_agent": "workflow_complete",
                "workflow_status": "completed",
                "workspace_files": state.get('workspace_files', []) + ['business_recommendations.md'],
                "messages": state.get('messages', []) + [AIMessage(content="✅ Business translation completed")],
                "completed_tasks_count": completed_count,
                "total_tasks_count": total_count,
                "project_progress": progress
            }
            
        except Exception as e:
            print(f"## Business Translator error: {str(e)}")
            return {
                **state,
                "workflow_status": "failed",
                "messages": state.get('messages', []) + [AIMessage(content=f"❌ Business Translator failed: {str(e)}")]
            }
    
    def finalize_workflow(self, state):
        """Finalize workflow and generate summary"""
        print("# Finalizing workflow")
        
        completed_tasks = state.get('completed_tasks', [])
        workspace_files = state.get('workspace_files', [])
        
        summary = f"""
🎉 Multi-Agent Workflow Completed Successfully!

**Completed Tasks**: {len(completed_tasks)}/5
- {', '.join(completed_tasks)}

**Generated Files**: {len(workspace_files)}
- {', '.join(workspace_files)}

**Session ID**: {state.get('session_id', 'default_session')}
**Duration**: Workflow completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

All deliverables have been saved to the workspace directory.
        """
        
        return {
            **state,
            "workflow_status": "finalized",
            "messages": state.get('messages', []) + [AIMessage(content=summary)]
        }
    
    def should_continue(self, state):
        """Decide whether to continue to next agent or end"""
        current_agent = state.get('current_agent', '')
        workflow_status = state.get('workflow_status', '')
        
        if workflow_status == 'failed':
            return "end"
        elif workflow_status == 'completed':
            return "finalize"
        elif current_agent == "workflow_complete":
            return "finalize"
        else:
            return "continue"
    
    def route_to_next_agent(self, state):
        """Route to the next agent based on current state"""
        current_agent = state.get('current_agent', '')
        
        agent_flow = {
            "business_analyst": "project_manager",
            "project_manager": "data_analyst", 
            "data_analyst": "ml_engineer",
            "ml_engineer": "business_translator",
            "business_translator": "finalize"
        }
        
        return agent_flow.get(current_agent, "end")
    