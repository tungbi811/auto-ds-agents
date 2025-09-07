from typing import Dict, List, Any, TypedDict
import uuid
from datetime import datetime

class DataScienceState(TypedDict, total=False):
    """State container for the data science workflow"""
    session_id: str
    user_request: str
    dataset_path: str
    current_phase: str
    current_agent: str
    next_action: str
    completed_tasks: List[str]
    data_profile: Dict[str, Any]
    model_results: Dict[str, Any]
    business_recommendations: List[str]
    code_history: List[Dict[str, Any]]
    workspace_files: Dict[str, str]
    errors: List[str]
    search_results: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    final_response: str
    iteration_count: int
    last_updated: str

def create_initial_state(user_request: str, dataset_path: str = "") -> DataScienceState:
    """Create initial state for workflow execution"""
    return DataScienceState(
        session_id=str(uuid.uuid4()),
        user_request=user_request,
        dataset_path=dataset_path,
        current_phase="start",  # Changed from "initialization"
        current_agent="project_manager",
        next_action="analyze_request",
        completed_tasks=[],
        data_profile={},
        model_results={},
        business_recommendations=[],
        code_history=[],
        workspace_files={},
        errors=[],
        search_results=[],
        analysis_results={},
        final_response="",
        iteration_count=0,
        last_updated=datetime.now().isoformat()
    )

def update_state_timestamp(state: DataScienceState) -> DataScienceState:
    """Update the state timestamp"""
    state['last_updated'] = datetime.now().isoformat()
    return state