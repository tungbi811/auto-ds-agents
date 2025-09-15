from dotenv import load_dotenv

from crew.crew_tools import set_session_id
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import DataScienceState
from .nodes import DataScienceNodes

class DataScienceWorkflow:
    """
    LangGraph workflow that orchestrates CrewAI agents
    Following the exact pattern of your existing WorkFlow class
    """
    
    def __init__(self):
        self.nodes = DataScienceNodes()
        self.memory = MemorySaver()
        self.app = self._build_workflow()
    
    def _build_workflow(self):
        """Build the workflow graph"""
        workflow = StateGraph(DataScienceState)
        
        # Add nodes (each wraps a CrewAI agent)
        workflow.add_node("load_requirements", self.nodes.load_requirements)
        workflow.add_node("business_analyst", self.nodes.business_analyst_node)
        workflow.add_node("project_manager", self.nodes.project_manager_node)
        workflow.add_node("data_analyst", self.nodes.data_analyst_node)
        workflow.add_node("ml_engineer", self.nodes.ml_engineer_node)
        workflow.add_node("business_translator", self.nodes.business_translator_node)
        workflow.add_node("finalize", self.nodes.finalize_workflow)
        
        # Set entry point
        workflow.add_edge(START, "load_requirements")
        
        # Add conditional edges for flow control
        workflow.add_conditional_edges(
            "load_requirements",
            self.nodes.should_continue,
            {
                "continue": "business_analyst",
                "end": END,
                "finalize": "finalize"
            }
        )
        
        # Sequential flow between agents
        workflow.add_conditional_edges(
            "business_analyst", 
            self.nodes.should_continue,
            {
                "continue": "project_manager",
                "end": END,
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "project_manager",
            self.nodes.should_continue, 
            {
                "continue": "data_analyst",
                "end": END,
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "data_analyst",
            self.nodes.should_continue,
            {
                "continue": "ml_engineer", 
                "end": END,
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "ml_engineer",
            self.nodes.should_continue,
            {
                "continue": "business_translator",
                "end": END, 
                "finalize": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "business_translator",
            self.nodes.should_continue,
            {
                "continue": "finalize",
                "end": END,
                "finalize": "finalize"
            }
        )
        
        # Finalize always ends
        workflow.add_edge("finalize", END)
        
        # Compile with memory
        return workflow.compile(checkpointer=self.memory)

    def execute(self, user_request: str = "", dataset_path: str = None, requirements: dict = None, session_id: str = None):
        """
        Execute the workflow with requirements from chatbot
        """
        if session_id:
            set_session_id(session_id)
        else:
            session_id = "default_session"
            set_session_id(session_id)
        
        print(f"\n🚀 Starting LangGraph Multi-Agent Workflow")
        print(f"📝 User Request: {user_request}")
        print(f"📊 Dataset Path: {dataset_path}")
        print(f"{'='*60}")
        
        # Initial state
        initial_state = {
            "requirements": requirements or {},
            "messages": [],
            "current_agent": "",
            "completed_tasks": [],
            "workflow_status": "initializing", 
            "business_analysis": {},
            "project_plan": {},
            "data_analysis": {},
            "model_results": {},
            "business_recommendations": {},
            "workspace_files": [],
            "session_id": session_id or "",
            "dataset_path": dataset_path or "", 
            "dataset_info": {},
            "code_history": []
        }
        
        try:
            # Execute the workflow with required checkpointer config
            thread_id = requirements.get('session_id', 'main_thread') if requirements else 'main_thread'
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config=config)
            
            print("✅ LangGraph workflow completed successfully!")
            print(f"📊 Final Status: {result.get('workflow_status')}")
            print(f"📁 Files Generated: {len(result.get('workspace_files', []))}")
            
            return {
                "success": True,
                "result": result.get('messages', [])[-1].content if result.get('messages') else "Workflow completed",
                "state": result,
                "session_id": result.get('session_id'),
                "completed_tasks": result.get('completed_tasks', []),
                "workspace_files": result.get('workspace_files', []),
                "workflow_status": result.get('workflow_status')
            }
            
        except Exception as e:
            error_msg = f"❌ LangGraph workflow execution failed: {str(e)}"
            print(error_msg)
            
            import traceback
            print("\n🔧 Detailed error trace:")
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "result": None,
                "state": initial_state
            }

    def stream_execute(self, user_request: str = "", dataset_path: str = None, requirements: dict = None, session_id: str = None):
        """
        Stream the workflow execution for real-time updates
        """
        initial_state = {
            "requirements": requirements or {},
            "messages": [],
            "current_agent": "",
            "completed_tasks": [],
            "workflow_status": "initializing", 
            "business_analysis": {},
            "project_plan": {},
            "data_analysis": {},
            "model_results": {},
            "business_recommendations": {},
            "workspace_files": [],
            "session_id": session_id or "",
            "dataset_path": dataset_path or "",  # This line already exists
            "dataset_info": {},
            "code_history": []
        }
        
        # Stream execution with required checkpointer config
        thread_id = requirements.get('session_id', 'main_thread') if requirements else 'main_thread'
        config = {"configurable": {"thread_id": thread_id}}
        for output in self.app.stream(initial_state, config=config):
            yield output