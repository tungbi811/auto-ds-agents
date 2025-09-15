import streamlit as st
import json
import os
import traceback
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from utils.file_handler import dataset_upload_section
from workflow.graph import DataScienceWorkflow


st.set_page_config(
    page_title="Multi-Agent Data Science System",
    page_icon="🤖",
    layout="wide"
)

def load_saved_requirements():
    """Load all saved requirements files"""
    workspace_dir = Path("workspace")
    if not workspace_dir.exists():
        return []
    
    requirements_files = []
    for file in workspace_dir.glob("requirements_*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                requirements_files.append(data)
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    
    return sorted(requirements_files, key=lambda x: x.get('timestamp', ''), reverse=True)

def display_execution_logs(state_history):
    """Display detailed execution logs"""
    if not state_history:
        return
    
    st.subheader("🔍 Execution Logs")
    
    with st.expander("View Detailed Agent Execution", expanded=False):
        # Get the final state with all code history
        final_state = state_history[-1] if state_history else {}
        code_history = final_state.get('code_history', [])
        
        if not code_history:
            st.write("No code execution history found")
            return
        
        # Display each agent's code execution
        for i, execution in enumerate(code_history):
            agent = execution.get('agent', 'Unknown')
            timestamp = execution.get('timestamp', 'Unknown time')
            success = execution.get('success', False)
            code = execution.get('code', 'No code')
            output = execution.get('output', 'No output')
            error = execution.get('error', '')
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = "Unknown"
            
            status_icon = "✅" if success else "❌"
            
            st.markdown(f"**Step {i+1}** - `{time_str}` - **{agent}** {status_icon}")
            
            # Show code in expandable section
            with st.expander(f"Code executed by {agent}", expanded=False):
                st.code(code, language='python')
                
                # Show output if available
                if output:
                    st.markdown("**Output:**")
                    st.text(output[:500] + "..." if len(output) > 500 else output)
                
                # Show errors if any
                if error:
                    st.markdown("**Error:**")
                    st.error(error[:300] + "..." if len(error) > 300 else error)
            
            st.markdown("---")
        
        # Show completed tasks summary
        completed_tasks = final_state.get('completed_tasks', [])
        if completed_tasks:
            st.markdown(f"**✅ Completed Tasks**: {', '.join(completed_tasks)}")

def main():
    st.title("🤖 Multi-Agent Data Science Workflow System")
    
    # Initialize session state
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = {}
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🚀 Workflow Selection")
        
        workflow_choice = st.radio(
            "Choose your workflow:",
            [
                "📋 Information Gathering",
                "📁 Dataset Upload",
                "⚙️ Agent Execution",
                "📊 View Saved Requirements"
            ],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📁 Workspace Status")
        
        # Check for saved requirements
        saved_reqs = load_saved_requirements()
        if saved_reqs:
            st.success(f"✅ {len(saved_reqs)} saved requirement(s)")
            for req in saved_reqs[:3]:  # Show latest 3
                with st.expander(f"📄 {req['session_id'][:8]}...", expanded=False):
                    st.text(f"Created: {req.get('timestamp', 'Unknown')[:16]}")
                    if 'requirements' in req:
                        st.text(f"Problem: {req['requirements'].get('business_problem', 'N/A')[:50]}...")
        else:
            st.info("ℹ️ No saved requirements yet")
        
        # Check for uploaded datasets
        st.markdown("---")
        st.markdown("### 📊 Available Datasets")
        
        workspace_dir = Path("workspace")
        if workspace_dir.exists():
            dataset_files = []
            for ext in ['*.csv', '*.xlsx', '*.parquet']:
                dataset_files.extend(workspace_dir.glob(ext))
            
            if dataset_files:
                st.success(f"✅ {len(dataset_files)} dataset(s) available")
                for dataset in dataset_files[:3]:  # Show latest 3
                    file_size = dataset.stat().st_size / 1024  # KB
                    st.text(f"📊 {dataset.name} ({file_size:.1f} KB)")
            else:
                st.info("ℹ️ No datasets uploaded yet")
        else:
            st.info("ℹ️ No workspace directory")
    
    # Main content area
    if workflow_choice == "📋 Information Gathering":
        st.markdown("""
        ## Step 1: Information Gathering Chatbot
        
        Start by defining your data science project requirements through our interactive chatbot.
        This will collect all necessary information before proceeding to agent execution.
        """)
        
        if st.button("🚀 Launch Information Gathering Chatbot", type="primary", use_container_width=True):
            st.switch_page("pages/chatbot.py")  
        
        # Show example workflow
        with st.expander("💡 What information will be collected?", expanded=True):
            st.markdown("""
            The chatbot will gather:
            
            - **Business Problem**: What you're trying to solve
            - **Success Metrics**: How to measure success
            - **Available Data**: What datasets you have
            - **Timeline**: Project deadlines and milestones
            - **Stakeholders**: Who's involved in the project
            - **Problem Type**: Classification, regression, clustering, etc.
            - **Constraints**: Any limitations or requirements
            - **Additional Context**: Any other relevant information
            """)
    
    elif workflow_choice == "📁 Dataset Upload":
        file_info = dataset_upload_section()
        
        if file_info:
            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Proceed to Agent Execution", type="primary"):
                    st.success("Dataset ready! Switch to 'Agent Execution' tab.")
            with col2:
                if st.button("🗑️ Remove Dataset"):
                    try:
                        os.remove(file_info['path'])
                        st.success("Dataset removed")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    elif workflow_choice == "⚙️ Agent Execution":
        st.markdown("""
        ## Step 2: Multi-Agent Execution Workflow
        
        Execute your data science project using our specialized AI agents.
        """)
        
        saved_reqs = load_saved_requirements()
        if not saved_reqs:
            st.warning("⚠️ No saved requirements found. Please complete information gathering first.")
            if st.button("👈 Go to Information Gathering"):
                st.rerun()
        else:
            # Let user select which requirements to use
            st.markdown("### 📋 Select Requirements")
            
            options = []
            for req in saved_reqs:
                problem = req.get('requirements', {}).get('business_problem', 'Unknown Problem')
                timestamp = req.get('timestamp', '')[:16]
                options.append(f"{problem[:50]}... ({timestamp})")
            
            selected_idx = st.selectbox("Choose requirements to execute:", range(len(options)), format_func=lambda x: options[x])
            selected_req = saved_reqs[selected_idx]
            
            # Display selected requirements
            st.markdown("### 📊 Selected Requirements")
            with st.expander("View Details", expanded=True):
                req_data = selected_req.get('requirements', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Problem**: {req_data.get('business_problem', 'N/A')}")
                    st.write(f"**Type**: {req_data.get('problem_type', 'N/A')}")
                    st.write(f"**Timeline**: {req_data.get('timeline', 'N/A')}")
                
                with col2:
                    st.write(f"**Stakeholders**: {', '.join(req_data.get('stakeholders', []))}")
                    st.write(f"**Success Metrics**: {', '.join(req_data.get('success_metrics', []))}")
            
            # Launch execution options
            col1, col2 = st.columns([2, 1])
            with col1:
                launch_execution = st.button("🚀 Launch Multi-Agent Execution", type="primary", use_container_width=True)
            with col2:
                detailed_logs = st.checkbox("Show detailed logs", value=False)
            
            if launch_execution:
                st.success("🎉 Launching multi-agent workflow with selected requirements!")
                
                # Create progress placeholder
                progress_placeholder = st.empty()
                
                # Execute with LangGraph workflow
                try:
                    # Initialize the workflow
                    workflow = DataScienceWorkflow()
                    
                    # Execute with selected requirements
                    req_data = selected_req.get('requirements', {})
                    
                    # Use progress bar for workflow execution
                    with progress_placeholder.container():
                        st.markdown("### 🚀 Workflow Execution Progress")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Execute the workflow
                        status_text.text("Initializing workflow...")
                        progress_bar.progress(10)
                        
                        result = workflow.execute(
                            user_request=req_data.get('business_problem', 'Data Science Analysis'),
                            requirements=req_data,
                            dataset_path=dataset_files[0] if dataset_files else None,
                            session_id=selected_req.get('session_id', 'default_session')
                        )
                        
                        # Store execution state
                        st.session_state.execution_history = [result.get('state', {})]
                        st.session_state.workflow_state = result.get('state', {})
                        
                        progress_bar.progress(100)
                        status_text.text("Workflow completed!")
                    
                    if result.get('success'):
                        st.success("✅ Workflow completed successfully!")
                        
                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Tasks Completed", len(result.get('completed_tasks', [])))
                        with col2:
                            st.metric("Files Generated", len(result.get('workspace_files', [])))
                        with col3:
                            st.metric("Status", result.get('workflow_status', 'Unknown'))
                        
                        # Show final result
                        st.markdown("### 📋 Execution Results")
                        st.write(result.get('result', 'No result'))
                        
                        # Show generated files
                        if result.get('workspace_files'):
                            st.markdown("### 📁 Generated Files")
                            for file in result['workspace_files']:
                                st.write(f"• {file}")
                                
                        # Show execution logs if requested
                        if detailed_logs and st.session_state.execution_history:
                            display_execution_logs(st.session_state.execution_history)
                    
                        # Workflow state inspector
                        if st.session_state.workflow_state and detailed_logs:
                            st.markdown("### 🔬 Workflow State Inspector")
                            with st.expander("Current Workflow State", expanded=False):
                                st.json(st.session_state.workflow_state)
                                
                    else:
                        st.error(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Execution error: {str(e)}")
                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())
    
    elif workflow_choice == "📊 View Saved Requirements":
        st.markdown("## 📊 Saved Requirements")
        
        saved_reqs = load_saved_requirements()
        if not saved_reqs:
            st.info("No saved requirements found.")
        else:
            for i, req in enumerate(saved_reqs):
                with st.expander(f"📄 Requirement #{i+1} - {req.get('session_id', 'Unknown')[:8]}...", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Created**: {req.get('timestamp', 'Unknown')}")
                        st.markdown(f"**Status**: {req.get('status', 'Unknown')}")
                        
                        if 'requirements' in req:
                            req_data = req['requirements']
                            st.markdown("---")
                            st.markdown("**Requirements Details:**")
                            st.json(req_data)
                    
                    with col2:
                        st.markdown("**Actions**")
                        if st.button(f"🚀 Execute", key=f"exec_{i}"):
                            st.success("Would launch execution with this requirement set")
                        
                        if st.button(f"🗑️ Delete", key=f"del_{i}"):
                            try:
                                os.remove(f"workspace/{req.get('filename', '')}")
                                st.success("Deleted successfully")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting: {str(e)}")

if __name__ == "__main__":
    main()