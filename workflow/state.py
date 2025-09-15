from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph.message import add_messages

class DataScienceState(TypedDict):
    """
    State for the data science multi-agent workflow
    Following the same pattern as your existing EmailsState
    """
    # Requirements from chatbot
    requirements: Dict[str, Any]
    
    # Messages for communication
    messages: Annotated[list, add_messages]
    
    # Workflow tracking
    current_agent: str
    completed_tasks: List[str]
    workflow_status: str  # 'initializing', 'running', 'completed', 'failed'
    
    # Agent results
    business_analysis: Dict[str, Any]
    project_plan: Dict[str, Any] 
    data_analysis: Dict[str, Any]
    model_results: Dict[str, Any]
    business_recommendations: Dict[str, Any]
    
    # File paths for workspace
    workspace_files: List[str]
    session_id: str
    
    # Todo management
    todo_tasks: List[Dict[str, Any]]
    completed_tasks_count: int
    total_tasks_count: int
    project_progress: int  # percentage
    
    # Dataset information
    dataset_path: str
    dataset_info: Dict[str, Any]
    
    # Code execution history
    code_history: List[Dict[str, Any]]