from typing import Dict, Any
from core.state import DataScienceState, update_state_timestamp

class ProjectManagerAgent:
    """Orchestrates workflow and manages phase transitions"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.name = "ProjectManager"
    
    def __call__(self, state: DataScienceState) -> DataScienceState:
        """Manage workflow transitions and coordination"""
        # Track iterations to prevent infinite loops
        iteration_count = state.get('iteration_count', 0) + 1
        state['iteration_count'] = iteration_count
        
        # Force termination after max iterations
        if iteration_count > 10:
            state['next_action'] = 'complete'
            state['current_phase'] = 'completed'
            return update_state_timestamp(state)
        
        current_phase = state.get('current_phase', 'start')
        completed_tasks = state.get('completed_tasks', [])
        
        # Phase transition logic
        if current_phase == 'start':
            state['current_phase'] = 'data_understanding'
            state['next_action'] = 'data_analyst' 
        
        elif current_phase == 'data_understanding':
            if 'data_profiling' in completed_tasks:
                state['current_phase'] = 'data_preparation'
                state['next_action'] = 'data_analyst'  # Stay with data analyst
            else:
                state['next_action'] = 'data_analyst'  # Continue with data analyst
        
        elif current_phase == 'data_preparation':
            if 'data_preprocessing' in completed_tasks:
                state['current_phase'] = 'modeling'
                state['next_action'] = 'ml_engineer'
            else:
                state['next_action'] = 'data_analyst'  # Continue with data analyst
        
        elif current_phase == 'modeling':
            if 'model_building' in completed_tasks:
                state['current_phase'] = 'business_translation'
                state['next_action'] = 'business_translator'
            else:
                state['next_action'] = 'ml_engineer'  # Continue with ML engineer
        
        elif current_phase == 'business_translation':
            if 'business_translation' in completed_tasks:
                state['next_action'] = 'complete'
                state['current_phase'] = 'completed'
            else:
                state['next_action'] = 'business_translator'  # Continue with translator
        
        else:
            # Fallback termination
            state['next_action'] = 'complete'
            state['current_phase'] = 'completed'
        
        return update_state_timestamp(state)