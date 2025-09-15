import streamlit as st
import os
import json
import uuid
from datetime import datetime
from typing import List, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

# Page configuration
st.set_page_config(
    page_title="Data Science Requirements Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize OpenAI
if 'openai_client' not in st.session_state:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please set OPENAI_API_KEY in your environment variables")
        st.stop()

# Data model for project requirements
class ProjectRequirements(BaseModel):
    """Data Science Project Requirements"""
    business_problem: str
    success_metrics: List[str]
    available_data: str
    timeline: str
    stakeholders: List[str]
    problem_type: str  # classification, regression, clustering, etc.
    constraints: List[str]
    additional_notes: str

# System template for information gathering
info_template = """You are a Business Analyst helping gather requirements for a data science project.

Your job is to get the following information from the user:

- What is the main business problem they want to solve?
- What success metrics will define project success?
- What data is available (describe datasets, sources, quality)?
- What is the expected timeline for the project?
- Who are the key stakeholders involved?
- What type of problem is this (classification, regression, clustering, forecasting, etc.)?
- Are there any constraints or limitations we should be aware of?
- Any additional notes or context?

Be friendly, professional, and ask clarifying questions if needed. Do not guess or assume information.

When you have gathered all the necessary information, call the ProjectRequirements tool to finalize the requirements."""

def get_info_messages(messages):
    return [SystemMessage(content=info_template)] + messages

# State definition
class State(TypedDict):
    messages: Annotated[list, add_messages]
    requirements: dict

# Initialize LLM and workflow components
@st.cache_resource
def initialize_chatbot():
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    llm_with_tool = llm.bind_tools([ProjectRequirements])
    
    def info_chain(state):
        messages = get_info_messages(state["messages"])
        response = llm_with_tool.invoke(messages)
        return {"messages": [response]}
    
    def get_state(state):
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return "save_requirements"
        elif not isinstance(messages[-1], HumanMessage):
            return END
        return "info"
    
    @st.cache_data
    def save_requirements_to_file(requirements_data, session_id):
        """Save requirements to workspace directory"""
        workspace_dir = "workspace"
        os.makedirs(workspace_dir, exist_ok=True)
        
        # Add metadata
        requirements_with_meta = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "requirements": requirements_data
        }
        
        filename = f"requirements_{session_id}.json"
        filepath = os.path.join(workspace_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(requirements_with_meta, f, indent=2)
        
        return filepath
    
    def save_requirements_node(state: State):
        if state["messages"][-1].tool_calls:
            requirements_data = state["messages"][-1].tool_calls[0]["args"]
            session_id = st.session_state.get('session_id', 'default')
            
            # Save to file
            filepath = save_requirements_to_file(requirements_data, session_id)
            
            return {
                "messages": [
                    ToolMessage(
                        content=f"Requirements saved successfully to {filepath}",
                        tool_call_id=state["messages"][-1].tool_calls[0]["id"],
                    )
                ],
                "requirements": requirements_data
            }
        return {"messages": [], "requirements": {}}
    
    # Build workflow
    memory = InMemorySaver()
    workflow = StateGraph(State)
    
    workflow.add_node("info", info_chain)
    workflow.add_node("save_requirements", save_requirements_node)
    
    workflow.add_conditional_edges("info", get_state, ["save_requirements", "info", END])
    workflow.add_edge("save_requirements", END)
    workflow.add_edge(START, "info")
    
    graph = workflow.compile(checkpointer=memory)
    return graph

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None

if 'requirements_completed' not in st.session_state:
    st.session_state.requirements_completed = False

if 'saved_requirements' not in st.session_state:
    st.session_state.saved_requirements = {}

# Main app
def main():
    st.title("🤖 Data Science Project Requirements Chatbot")
    st.markdown("""
    **Gather comprehensive requirements for your data science project**
    
    This chatbot will help you define:
    - Business problem and objectives
    - Success metrics and KPIs
    - Available data and sources
    - Timeline and stakeholders
    - Project constraints
    """)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Session Info")
        st.markdown(f"**Session ID**: `{st.session_state.session_id[:8]}...`")
        
        if st.session_state.requirements_completed:
            st.success("✅ Requirements Complete!")
            
            # Show requirements summary
            if st.session_state.saved_requirements:
                st.markdown("### 📊 Requirements Summary")
                with st.expander("View Details", expanded=True):
                    reqs = st.session_state.saved_requirements
                    st.write(f"**Problem**: {reqs.get('business_problem', 'N/A')}")
                    st.write(f"**Type**: {reqs.get('problem_type', 'N/A')}")
                    st.write(f"**Timeline**: {reqs.get('timeline', 'N/A')}")
                    st.write(f"**Stakeholders**: {', '.join(reqs.get('stakeholders', []))}")
            
            # Action buttons
            st.markdown("### 🚀 Next Steps")
            if st.button("Start Agent Execution", type="primary", use_container_width=True):
                st.success("Requirements passed to agent workflow!")
                st.balloons()
                # Here you would trigger your LangGraph multi-agent workflow
                
        else:
            st.info("💬 Gathering requirements...")
        
        # Reset button
        if st.button("🔄 Start New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Progress indicators
        st.markdown("### ✅ Information Checklist")
        checklist_items = [
            "Business Problem",
            "Success Metrics", 
            "Available Data",
            "Timeline",
            "Stakeholders",
            "Problem Type",
            "Constraints"
        ]
        
        for item in checklist_items:
            if st.session_state.requirements_completed:
                st.markdown(f"✅ {item}")
            else:
                st.markdown(f"⏳ {item}")
    
    # Initialize chatbot
    try:
        graph = initialize_chatbot()
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return
    
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show tool calls for debugging
            if message["role"] == "assistant" and "tool_calls" in message:
                with st.expander("🔧 Requirements Detected", expanded=False):
                    st.json(message["tool_calls"])
    
    # Example prompts
    if not st.session_state.messages:
        st.markdown("**💡 Example prompts to get started:**")
        example_cols = st.columns(3)
        
        with example_cols[0]:
            if st.button("🎯 Customer Churn Analysis"):
                example_prompt = "I want to predict which customers are likely to churn so we can take preventive action."
                st.session_state.example_prompt = example_prompt
                st.rerun()
        
        with example_cols[1]:
            if st.button("📈 Sales Forecasting"):
                example_prompt = "We need to forecast sales for the next quarter to optimize inventory and resource planning."
                st.session_state.example_prompt = example_prompt
                st.rerun()
        
        with example_cols[2]:
            if st.button("🔍 Fraud Detection"):
                example_prompt = "Help us detect fraudulent transactions in our payment system to reduce losses."
                st.session_state.example_prompt = example_prompt
                st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Describe your data science project or ask questions..."):
        process_user_input(prompt, graph)
    
    # Handle example prompts
    if hasattr(st.session_state, 'example_prompt'):
        process_user_input(st.session_state.example_prompt, graph)
        del st.session_state.example_prompt

def process_user_input(prompt, graph):
    """Process user input through the workflow"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process through workflow
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your requirements..."):
            try:
                config = {"configurable": {"thread_id": st.session_state.session_id}}
                
                # Stream the response
                response_placeholder = st.empty()
                
                for output in graph.stream(
                    {"messages": [HumanMessage(content=prompt)]}, 
                    config=config, 
                    stream_mode="updates"
                ):
                    if output:
                        for node_name, node_output in output.items():
                            if "messages" in node_output and node_output["messages"]:
                                last_message = node_output["messages"][-1]
                                
                                if isinstance(last_message, AIMessage):
                                    # Display AI response
                                    response_placeholder.markdown(last_message.content)
                                    
                                    # Add to session state
                                    message_dict = {
                                        "role": "assistant",
                                        "content": last_message.content
                                    }
                                    
                                    # Include tool calls if present
                                    if last_message.tool_calls:
                                        message_dict["tool_calls"] = last_message.tool_calls
                                    
                                    st.session_state.messages.append(message_dict)
                                
                                elif isinstance(last_message, ToolMessage):
                                    # Requirements completed
                                    st.success("✅ Requirements gathering complete!")
                                    st.session_state.requirements_completed = True
                                    
                                    # Save requirements to session state
                                    if "requirements" in node_output:
                                        st.session_state.saved_requirements = node_output["requirements"]
                                    
                                    # Show next steps
                                    st.markdown("""
                                    **🎉 Great! I've collected all the necessary information.**
                                    
                                    Your requirements have been saved and are ready for the multi-agent workflow.
                                    Click **"Start Agent Execution"** in the sidebar to begin the analysis.
                                    """)
                                    
                                    st.rerun()
                
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
                st.error("Please try again or start a new session.")

if __name__ == "__main__":
    main()