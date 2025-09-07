from langgraph.graph import StateGraph, END
from typing import Dict, Any

from core.state import DataScienceState, create_initial_state
from core.memory import WorkspaceManager, TodoManager
from agents.project_manager import ProjectManagerAgent
from agents.data_analyst import DataAnalystAgent
from agents.ml_engineer import MLEngineerAgent
from agents.business_translator import BusinessTranslatorAgent

class DataScienceWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self, llm, tools: Dict[str, Any]):
        self.llm = llm
        self.tools = tools
        self.workspace = WorkspaceManager()
        self.todo_manager = TodoManager(self.workspace)
        
        # Initialize agents
        self.manager = ProjectManagerAgent(llm, tools)
        self.data_analyst = DataAnalystAgent(llm, tools)
        self.ml_engineer = MLEngineerAgent(llm, tools)
        self.business_translator = BusinessTranslatorAgent(llm, tools)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(DataScienceState)
        
        # Add agent nodes
        workflow.add_node("manager", self.manager_node)
        workflow.add_node("data_analyst", self.data_analyst_node)
        workflow.add_node("ml_engineer", self.ml_engineer_node)
        workflow.add_node("business_translator", self.business_translator_node)
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "manager",
            self.route_next_agent,
            {
                "data_analyst": "data_analyst",
                "ml_engineer": "ml_engineer", 
                "business_translator": "business_translator",
                "complete": END
            }
        )
        
        # Route back to manager after each agent
        workflow.add_edge("data_analyst", "manager")
        workflow.add_edge("ml_engineer", "manager")
        workflow.add_edge("business_translator", "manager")
        
        # Set entry point
        workflow.set_entry_point("manager")
        
        return workflow.compile()
    
    def manager_node(self, state: DataScienceState) -> DataScienceState:
        """Execute project manager agent"""
        state = self._track_agent_execution(state, "manager")
        return self.manager(state)
    
    def data_analyst_node(self, state: DataScienceState) -> DataScienceState:
        """Execute data analyst agent"""
        state = self._track_agent_execution(state, "data_analyst")
        return self.data_analyst.execute_action(state)
    
    def ml_engineer_node(self, state: DataScienceState) -> DataScienceState:
        """Execute ML engineer agent"""
        state = self._track_agent_execution(state, "ml_engineer")
        return self.ml_engineer.execute_action(state)
    
    def business_translator_node(self, state: DataScienceState) -> DataScienceState:
        """Execute business translator agent"""
        state = self._track_agent_execution(state, "business_translator")
        return self.business_translator.execute_action(state)
    
    def _track_agent_execution(self, state: DataScienceState, agent_name: str) -> DataScienceState:
        """Track agent execution to prevent infinite loops"""
        if 'agent_execution_count' not in state:
            state['agent_execution_count'] = {}
        
        state['agent_execution_count'][agent_name] = state['agent_execution_count'].get(agent_name, 0) + 1
        state['current_agent'] = agent_name
        
        return state
    
    def route_next_agent(self, state: DataScienceState) -> str:
        """Determine which agent should execute next"""
        current_phase = state.get('current_phase', '')
        next_action = state.get('next_action', '')
        completed_tasks = state.get('completed_tasks', [])
        iteration_count = state.get('iteration_count', 0)
        
        # Force termination after too many iterations
        if iteration_count > 12:
            return "complete"
        
        # Check if workflow is complete
        if (next_action == 'complete' or 
            current_phase == 'completed' or
            'business_translation' in completed_tasks):
            return "complete"
        
        # Route based on next_action from manager
        if next_action == 'data_analyst':
            return "data_analyst"
        elif next_action == 'ml_engineer':
            return "ml_engineer" 
        elif next_action == 'business_translator':
            return "business_translator"
        
        # Fallback based on phase and completed tasks
        if current_phase in ['data_understanding', 'data_preparation']:
            if 'data_profiling' not in completed_tasks:
                return "data_analyst"
            elif 'data_preprocessing' not in completed_tasks:
                return "data_analyst"
        
        elif current_phase == 'modeling':
            if 'model_building' not in completed_tasks:
                return "ml_engineer"
        
        elif current_phase == 'business_translation':
            if 'business_translation' not in completed_tasks:
                return "business_translator"
        
        # Default termination
        return "complete"
    
    def execute(self, user_request: str, dataset_path: str = None) -> Dict[str, Any]:
        """Execute the complete workflow"""
        
        # Create initial state
        initial_state = create_initial_state(user_request, dataset_path or "")
        
        try:
            # Execute workflow with recursion limit
            config = {"recursion_limit": 100}
            
            # Execute workflow with config  
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Generate final report
            return self._generate_final_report(final_state)
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            return {
                "success": False,
                "error": error_msg,
                "response": error_msg,
                "search_used": False
            }
    
    def _generate_final_report(self, state: DataScienceState) -> Dict[str, Any]:
        """Generate final report from completed workflow"""
        
        # Check if workflow completed successfully
        completed_tasks = state.get('completed_tasks', [])
        required_tasks = ['data_profiling', 'model_building', 'business_translation']
        
        success = all(task in completed_tasks for task in required_tasks)
        
        if success:
            # Generate comprehensive response
            response_parts = []
            
            # Executive Summary
            response_parts.append("# Data Science Analysis Complete")
            response_parts.append("")
            
            # Business Recommendations
            recommendations = state.get('business_recommendations', [])
            if recommendations:
                response_parts.append("## 🎯 Business Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    response_parts.append(f"{i}. {rec}")
                response_parts.append("")
            
            # Technical Summary
            model_results = state.get('model_results', {})
            if model_results:
                response_parts.append("## 📊 Technical Results")
                best_model = model_results.get('best_model', 'Unknown')
                response_parts.append(f"**Best Model:** {best_model}")
                
                metrics = model_results.get('performance_metrics', {})
                if metrics:
                    response_parts.append("**Performance Metrics:**")
                    for metric, value in metrics.items():
                        response_parts.append(f"- {metric.title()}: {value:.3f}")
                response_parts.append("")
            
            # Data Quality Insights
            data_profile = state.get('data_profile', {})
            if data_profile:
                quality_score = data_profile.get('quality_score', 0)
                response_parts.append(f"## 📈 Data Quality Score: {quality_score:.2f}/1.0")
                
                issues = data_profile.get('issues', [])
                if issues:
                    response_parts.append("**Data Issues Addressed:**")
                    for issue in issues:
                        response_parts.append(f"- {issue.replace('_', ' ').title()}")
                response_parts.append("")
            
            # Execution Summary
            code_history = state.get('code_history', [])
            response_parts.append(f"## ⚙️ Execution Summary")
            response_parts.append(f"- **Code Operations:** {len(code_history)}")
            response_parts.append(f"- **Agents Used:** {len(set(exec['agent'] for exec in code_history))}")
            
            errors = state.get('errors', [])
            if errors:
                response_parts.append(f"- **Issues Encountered:** {len(errors)}")
            
            final_response = "\n".join(response_parts)
            
        else:
            # Partial completion
            final_response = f"""
# Analysis Partially Complete

**Completed Tasks:** {', '.join(completed_tasks)}
**Remaining Tasks:** {', '.join(set(required_tasks) - set(completed_tasks))}

The analysis made progress but did not complete all phases. Please review the errors and try again.
"""
        
        # Check if web search was used
        search_used = any(
            'search_web' in str(exec.get('code', '')) 
            for exec in state.get('code_history', [])
        )
        
        return {
            "success": success,
            "response": final_response,
            "search_used": search_used,
            "state": state,
            "code_history": state.get('code_history', []),
            "visualizations": []  # Could extract from workspace
        }

def create_codeact_workflow(llm, tools: Dict[str, Any]) -> DataScienceWorkflow:
    """Factory function to create the workflow"""
    return DataScienceWorkflow(llm, tools)