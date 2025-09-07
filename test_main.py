# main.py - Enhanced with LangGraph Multi-Agent Workflow
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import pandas as pd
import json
from datetime import datetime

# Import the LangGraph workflow
from agents.workflow import create_codeact_workflow

# Import existing tools
from tools.basic_eda import generate_eda_report
from tools.web_search import web_search
from tools.document_analyze import analyze_pdf_from_path
from tools.code_execute import CodeExecutor

st.set_page_config(
    page_title="LangGraph Multi-Agent Data Science System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        st.stop()
    return OpenAI(api_key=api_key)

@st.cache_resource
def initialize_workflow():
    """Initialize the LangGraph multi-agent workflow"""
    client = get_openai_client()
    tools = {
        'analyze_dataset': generate_eda_report,
        'search_web': web_search,
        'analyze_pdf': analyze_pdf_from_path,
        'execute_code': CodeExecutor().execute_code
    }
    return create_codeact_workflow(client, tools)

def display_workflow_diagram():
    """Display the workflow diagram in sidebar"""
    st.sidebar.markdown("### 🔄 Multi-Agent Workflow")
    
    workflow_steps = [
        "📋 **Project Manager**: Task analysis & planning",
        "📊 **Data Analyst**: EDA & data profiling", 
        "🤖 **ML Engineer**: Model building & validation",
        "💼 **Business Translator**: Insights & recommendations"
    ]
    
    for step in workflow_steps:
        st.sidebar.markdown(step)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Flow**: Manager → Analyst → Engineer → Translator → Manager")

def display_system_status():
    """Display system status and capabilities"""
    st.sidebar.markdown("### ⚙️ System Status")
    
    # Check API key
    api_status = "✅ Connected" if os.getenv("OPENAI_API_KEY") else "❌ Missing API Key"
    st.sidebar.markdown(f"**OpenAI API**: {api_status}")
    
    # Check tools
    st.sidebar.markdown("**Available Tools**:")
    tools_status = [
        "✅ Code Execution",
        "✅ Web Search", 
        "✅ PDF Analysis",
        "✅ Dataset Analysis"
    ]
    for tool in tools_status:
        st.sidebar.markdown(f"  {tool}")

def analyze_uploaded_file(uploaded_file):
    """Analyze uploaded file and return basic info"""
    if uploaded_file is None:
        return None
    
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Basic file analysis
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(tmp_path)
            file_info = {
                "type": "CSV Dataset",
                "path": tmp_path,
                "rows": len(df),
                "columns": len(df.columns),
                "size": f"{os.path.getsize(tmp_path) / 1024:.1f} KB",
                "columns_list": list(df.columns)[:10]  # First 10 columns
            }
        elif uploaded_file.name.endswith('.pdf'):
            file_info = {
                "type": "PDF Document", 
                "path": tmp_path,
                "size": f"{os.path.getsize(tmp_path) / 1024:.1f} KB"
            }
        else:
            file_info = {
                "type": "Unknown",
                "path": tmp_path,
                "size": f"{os.path.getsize(tmp_path) / 1024:.1f} KB"
            }
        
        return file_info
    except Exception as e:
        st.error(f"Error analyzing file: {str(e)}")
        return None

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
                from datetime import datetime
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
    st.title("🤖 LangGraph Multi-Agent Data Science System")
    st.markdown("""
    **Advanced AI agents collaborating through executable code to solve complex data science tasks**
    
    This system demonstrates the research principles from Manus and AutoKaggle, featuring:
    - **CodeAct paradigm**: Agents use executable Python code as their action mechanism
    - **Structured workflow**: LangGraph orchestrates agent coordination
    - **Specialized agents**: Each agent has domain expertise and shared workspace
    """)
    
    # Initialize components
    workflow = initialize_workflow()
    
    # Sidebar components
    display_workflow_diagram()
    display_system_status()
    
    # File upload section
    st.sidebar.markdown("### 📁 Data Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset or document", 
        type=['csv', 'pdf', 'txt'],
        help="CSV for analysis, PDF for document processing"
    )
    
    # File analysis
    file_info = None
    if uploaded_file:
        file_info = analyze_uploaded_file(uploaded_file)
        if file_info:
            st.sidebar.success(f"✅ {file_info['type']} loaded")
            st.sidebar.markdown(f"**Size**: {file_info['size']}")
            if file_info['type'] == "CSV Dataset":
                st.sidebar.markdown(f"**Shape**: {file_info['rows']} rows × {file_info['columns']} cols")
                if file_info['columns_list']:
                    with st.sidebar.expander("Preview columns"):
                        for col in file_info['columns_list']:
                            st.sidebar.text(f"• {col}")
    
    # Workflow options
    st.sidebar.markdown("### 🎯 Workflow Options")
    workflow_mode = st.sidebar.selectbox(
        "Select execution mode",
        ["Full Pipeline", "Data Analysis Only", "Modeling Only", "Business Translation Only"],
        help="Choose which parts of the workflow to execute"
    )
    
    detailed_logs = st.sidebar.checkbox("Show detailed execution logs", value=False)
    
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "execution_history" not in st.session_state:
        st.session_state.execution_history = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display any attachments or execution details
            if "details" in message and detailed_logs:
                with st.expander("Execution Details"):
                    st.json(message["details"])
    
    # Example prompts
    st.markdown("**💡 Example prompts:**")
    example_cols = st.columns(3)
    
    with example_cols[0]:
        if st.button("🔍 Analyze uploaded dataset"):
            if file_info:
                example_prompt = f"Perform a comprehensive analysis of the uploaded {file_info['type'].lower()}. Include data profiling, quality assessment, and initial insights."
                st.session_state.example_prompt = example_prompt
    
    with example_cols[1]:
        if st.button("🤖 Build predictive model"):
            example_prompt = "Build and evaluate machine learning models for this dataset. Compare different algorithms and provide performance metrics."
            st.session_state.example_prompt = example_prompt
    
    with example_cols[2]:
        if st.button("💼 Generate business insights"):
            example_prompt = "Translate the technical findings into actionable business recommendations with clear impact metrics."
            st.session_state.example_prompt = example_prompt
    
    # Chat input
    prompt = st.chat_input("Describe your data science task...")
    
    # Handle example prompt selection
    if hasattr(st.session_state, 'example_prompt'):
        prompt = st.session_state.example_prompt
        delattr(st.session_state, 'example_prompt')
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Execute workflow
        with st.chat_message("assistant"):
            with st.spinner("🔄 Multi-agent workflow executing..."):
                
                # Create columns for real-time updates
                status_col, progress_col = st.columns([3, 1])
                
                with status_col:
                    status_placeholder = st.empty()
                    status_placeholder.markdown("🚀 **Initializing agents...**")
                
                with progress_col:
                    progress_bar = st.progress(0)
                
                try:
                    # Execute workflow with file path if available
                    dataset_path = file_info['path'] if file_info and file_info['type'] == "CSV Dataset" else None
                    
                    # Update status during execution
                    status_placeholder.markdown("📋 **Project Manager analyzing task...**")
                    progress_bar.progress(25)
                    
                    result = workflow.execute(prompt, dataset_path)
                    
                    # Update final status
                    status_placeholder.markdown("✅ **Workflow completed successfully!**")
                    progress_bar.progress(100)
                    
                    # Display results
                    response = result.get("response", "No response generated")
                    st.markdown(response)
                    
                    # Store execution details
                    execution_details = {
                        "success": result.get("success", False),
                        "search_used": result.get("search_used", False),
                        "code_operations": len(result.get("code_history", [])),
                        "workflow_mode": workflow_mode,
                        "file_processed": file_info['type'] if file_info else None
                    }
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "details": execution_details
                    })
                    
                    # Store workflow state for inspection
                    st.session_state.workflow_state = result.get("state", {})
                    st.session_state.execution_history.append(result.get("state", {}))
                    
                    # Success metrics
                    if result.get("success"):
                        st.success("🎉 Multi-agent workflow completed successfully!")
                        
                        # Display key metrics
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("Agents Used", len(set(
                                exec.get('agent', 'Unknown') 
                                for exec in result.get("code_history", [])
                            )))
                        with metrics_cols[1]:
                            st.metric("Code Operations", len(result.get("code_history", [])))
                        with metrics_cols[2]:
                            st.metric("Web Searches", "Yes" if result.get("search_used") else "No")
                        with metrics_cols[3]:
                            st.metric("Workflow Status", "Complete" if result.get("success") else "Partial")
                    
                except Exception as e:
                    error_msg = f"❌ Workflow execution failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    # Debug information
                    with st.expander("🔧 Debug Information"):
                        st.text(f"Error type: {type(e).__name__}")
                        st.text(f"Error details: {str(e)}")
                        if hasattr(e, '__traceback__'):
                            import traceback
                            st.code(traceback.format_exc(), language='python')
        
        # Display execution logs if requested
        if detailed_logs and st.session_state.execution_history:
            display_execution_logs(st.session_state.execution_history)
    
    # Workflow state inspector
    if st.session_state.workflow_state and detailed_logs:
        st.markdown("### 🔬 Workflow State Inspector")
        with st.expander("Current Workflow State", expanded=False):
            st.json(st.session_state.workflow_state)
    
    # Cleanup temporary files
    if uploaded_file and file_info:
        # Clean up temp file when session ends
        temp_path = file_info.get('path')
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass  # File might already be cleaned up
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🔬 <strong>Research Implementation</strong>: This system demonstrates multi-agent coordination 
    using CodeAct paradigm and LangGraph orchestration based on Manus and AutoKaggle research
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()