import streamlit as st
from autogen import UserProxyAgent
from multi_agents import (
    BusinessAnalyst, DataExplorer, DataCleaner, 
    FeatureEngineer, Coder, DataEngineer, 
    Modeler, Evaluator, BusinessTranslator
)
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, RevertToUserTarget
from autogen.agentchat.group.patterns import DefaultPattern
from dotenv import load_dotenv
import os
import tempfile
import time
from datetime import datetime
import threading
import sys
import io
import re
import queue

load_dotenv()


# Streamlit page config
st.set_page_config(
    page_title="AG2 Studio - Multi-Agent Data Science",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "workflow_complete" not in st.session_state:
    st.session_state.workflow_complete = False
if "workflow_thread" not in st.session_state:
    st.session_state.workflow_thread = None
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()

# Function to run workflow in background
def run_workflow_background(data_paths, user_query, max_rounds, msg_queue):
    """Run workflow in background thread"""
    # Redirect stdout to capture agent conversations
    original_stdout = sys.stdout
    captured_output = io.StringIO()
    
    # Custom stdout that captures and parses in real-time
    class RealTimeCapture:
        def __init__(self, original, queue):
            self.original = original
            self.queue = queue
            self.buffer = ""
            self.current_agent = None
            self.current_content = []
            self.in_clarification_call = False
            
        def write(self, text):
            # Write to original for debugging
            if self.original:
                self.original.write(text)
            
            # Process text
            if text:
                self.buffer += text
                # Process complete lines
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    for line in lines[:-1]:
                        self.process_line(line)
                    self.buffer = lines[-1]
            return len(text) if text else 0
            
        def process_line(self, line):
            line = line.strip()
            if not line:
                return
                
            # Detect "Next speaker:"
            if line.startswith("Next speaker:"):
                speaker = line.replace("Next speaker:", "").strip()
                if speaker != "_Group_Tool_Executor":
                    self.queue.put({
                        "role": "system",
                        "content": f"Next speaker: {speaker}",
                        "is_next_speaker": True,
                        "speaker": speaker,  # Add speaker field for detection
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
            
            # Detect request_clarification function calls
            elif "***** Suggested tool call" in line and "request_clarification" in line:
                # Mark that we're starting to capture a clarification request
                self.in_clarification_call = True
                
            # Capture the clarification question from Arguments section
            elif hasattr(self, 'in_clarification_call') and self.in_clarification_call and "clarification_question" in line:
                # Extract the question from the arguments
                question_match = re.search(r'"clarification_question":\s*"([^"]+)"', line)
                if question_match:
                    question = question_match.group(1)
                    self.queue.put({
                        "role": "assistant",
                        "sender": "BusinessAnalyst",
                        "recipient": "User", 
                        "content": f"🤔 **Question:** {question}",
                        "is_clarification": True,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    self.in_clarification_call = False
                    
            # Detect agent conversation
            elif " (to " in line and "):" in line and not line.startswith("*****"):
                match = re.match(r'^([^(]+) \(to ([^)]+)\):$', line)
                if match:
                    agent = match.group(1)
                    recipient = match.group(2)
                    if agent != "_Group_Tool_Executor":
                        # Save previous conversation
                        if self.current_agent and self.current_content:
                            content = '\n'.join(self.current_content)
                            self.queue.put({
                                "role": "assistant",
                                "sender": self.current_agent,
                                "recipient": "chat_manager",
                                "content": content,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                        # Start new conversation
                        self.current_agent = agent
                        self.current_content = []
                        
            # Detect separator
            elif "----------------" in line:
                # Save current conversation
                if self.current_agent and self.current_content:
                    content = '\n'.join(self.current_content)
                    self.queue.put({
                        "role": "assistant",
                        "sender": self.current_agent,
                        "recipient": "chat_manager",
                        "content": content,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    self.current_agent = None
                    self.current_content = []
                    
                self.queue.put({
                    "role": "separator",
                    "content": "---",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
            # Collect content
            elif self.current_agent:
                # Skip technical output
                if not any(line.startswith(x) for x in ["*****", ">>>>", "Call ID:", "Arguments:", "Input arguments:", "Output:"]):
                    self.current_content.append(line)
                    
            # Detect workflow termination and flush remaining content
            elif "TERMINATING RUN" in line:
                # Save current conversation before termination
                if self.current_agent and self.current_content:
                    content = '\n'.join(self.current_content)
                    self.queue.put({
                        "role": "assistant",
                        "sender": self.current_agent,
                        "recipient": "chat_manager",
                        "content": content,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    self.current_agent = None
                    self.current_content = []
                    
        def flush(self):
            if self.original:
                self.original.flush()
    
    try:
        # Set custom stdout
        sys.stdout = RealTimeCapture(original_stdout, msg_queue)
        
        # Initialize agents
        ba = BusinessAnalyst()
        data_explorer = DataExplorer()
        data_cleaner = DataCleaner()
        feature_engineer = FeatureEngineer()
        coder = Coder()
        data_engineer = DataEngineer()
        modeler = Modeler()
        evaluator = Evaluator()
        business_translator = BusinessTranslator()
        
        # Setup code execution config
        work_dir = tempfile.mkdtemp()
        code_exec_config = {
            "work_dir": work_dir,
            "use_docker": False,
            "timeout": 120,
            "last_n_messages": 3
        }
        
        # Copy dataset files to work directory so they're accessible to code execution
        accessible_data_paths = []
        for i, data_path in enumerate(data_paths):
            # Copy file to work directory
            filename = f"dataset_{i+1}.csv"
            accessible_path = os.path.join(work_dir, filename)
            import shutil
            shutil.copy2(data_path, accessible_path)
            accessible_data_paths.append(accessible_path)
        
        # Update data_paths to use accessible paths
        data_paths = accessible_data_paths
        
        # Custom UserProxyAgent that can get input from queue
        class StreamlitUserProxy(UserProxyAgent):
            def __init__(self, queue, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.input_queue = queue
                
            def get_human_input(self, prompt: str = "") -> str:
                # Signal UI that user input is needed
                self.input_queue.put({
                    "type": "user_input_needed",
                    "prompt": prompt
                })
                
                # Wait for user input from queue
                while True:
                    try:
                        message = self.input_queue.get(timeout=1)
                        if message.get("type") == "user_input":
                            return message.get("content", "")
                    except:
                        continue  # Keep waiting
        
        user = StreamlitUserProxy(
            queue=msg_queue,
            name="User",
            human_input_mode="ALWAYS",  # Changed to ALWAYS to allow interaction
            max_consecutive_auto_reply=0,
            code_execution_config=code_exec_config
        )
        
        # Set up handoffs
        ba.handoffs.set_after_work(AgentTarget(data_explorer))
        data_explorer.handoffs.set_after_work(AgentTarget(data_cleaner))
        data_cleaner.handoffs.set_after_work(AgentTarget(feature_engineer))
        feature_engineer.handoffs.set_after_work(AgentTarget(modeler))
        modeler.handoffs.set_after_work(AgentTarget(business_translator))
        business_translator.handoffs.set_after_work(RevertToUserTarget())
        
        pattern = DefaultPattern(
            initial_agent=ba,
            agents=[ba, data_explorer, data_cleaner, feature_engineer, 
                   coder, data_engineer, modeler, evaluator, business_translator],
            user_agent=user,
            group_manager_args=None,
        )
        
        # Prepare message
        if len(data_paths) == 1:
            full_message = f"Dataset path: {data_paths[0]}\n{user_query}"
        else:
            paths_str = "\n".join([f"Dataset {i+1}: {path}" for i, path in enumerate(data_paths)])
            full_message = f"Dataset paths:\n{paths_str}\n\n{user_query}"
        
        # Run workflow
        result, final_context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages=full_message,
            max_rounds=max_rounds
        )
        
        # Flush any remaining conversation content after workflow completion
        if hasattr(sys.stdout, 'current_agent') and sys.stdout.current_agent and sys.stdout.current_content:
            content = '\n'.join(sys.stdout.current_content)
            msg_queue.put({
                "role": "assistant",
                "sender": sys.stdout.current_agent,
                "recipient": "chat_manager",
                "content": content,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
    except Exception as e:
        msg_queue.put({
            "role": "system",
            "content": f"Error: {str(e)}",
            "is_status": True,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        # Signal completion
        msg_queue.put({"type": "workflow_complete"})

# UI Layout
st.title("Minion Studio")
st.caption("Multi-Agent Data Science Workflows with Live Interaction")

# Sidebar
with st.sidebar:
    st.header("Data Configuration")
    
    uploaded_files = st.file_uploader(
        "Upload Dataset(s)",
        type=["csv", "xlsx", "json"],
        accept_multiple_files=True,
        help="Upload one or more datasets for analysis"
    )
    
    data_paths = []
    if uploaded_files:
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_paths.append(file_path)
        
        if len(data_paths) == 1:
            st.success(f"{uploaded_files[0].name} uploaded")
        else:
            st.success(f"{len(data_paths)} files uploaded")
    
    st.divider()
    
    max_rounds = st.slider(
        "Max Conversation Rounds",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    st.divider()
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.processing = False
        st.session_state.workflow_complete = False
        st.rerun()

# Main content
col1, col2 = st.columns([7, 3])

with col1:
    st.subheader("Conversation")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        for message in st.session_state.messages:
            if message.get("is_status"):
                st.info(message["content"])
            elif message.get("is_next_speaker"):
                st.markdown(f"**{message['content']}**")
                st.markdown("")
            elif message.get("is_clarification"):
                # Special highlighting for clarification requests
                st.warning(f"🤔 **Agent Question:** {message['content'].replace('🤔 **Question:** ', '')}")
            elif message["role"] == "separator":
                st.markdown("---")
                st.markdown("")
            elif message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                agent_name = message.get("sender", "Assistant")
                recipient = message.get("recipient")
                
                with st.chat_message("assistant", avatar="🤖"):
                    if agent_name != "Assistant" and recipient:
                        st.markdown(f"**{agent_name} (to {recipient}):**")
                    st.markdown("")
                    st.markdown(message["content"])

# Check if workflow is expecting user input (define this early)
workflow_waiting_for_user = False
if st.session_state.messages:
    # Check last few messages for "Next speaker: User" or BusinessTranslator asking for decisions
    last_messages = st.session_state.messages[-5:]  # Check more messages
    for msg in last_messages:
        content = msg.get("content", "")
        if (content.startswith("Next speaker: User") or 
            msg.get("speaker") == "User" or
            ("tell me which" in content.lower() and msg.get("sender") == "BusinessTranslator") or
            ("pick any" in content.lower() and msg.get("sender") == "BusinessTranslator")):
            workflow_waiting_for_user = True
            break

with col2:
    st.subheader("Workflow Status")
    
    if st.session_state.processing:
        st.warning("Processing...")
        st.progress(0.5)
        
        # Debug: Show if waiting for user input
        if workflow_waiting_for_user:
            st.info("🔴 Waiting for user input")
        else:
            st.info("Agents are working")
            
    elif st.session_state.workflow_complete:
        st.success("Workflow Complete!")
    else:
        st.info("Ready to start")
        
    # Debug: Show last few messages
    if st.session_state.messages:
        st.write("**Debug - Last Messages:**")
        for msg in st.session_state.messages[-3:]:
            if msg.get("is_next_speaker"):
                st.write(f"- {msg.get('content')} (speaker: {msg.get('speaker')})")
            elif msg.get("content"):
                content_preview = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
                st.write(f"- {msg.get('role')}: {content_preview}")

# Input area
st.divider()

# Show appropriate input field
if workflow_waiting_for_user and st.session_state.processing:
    # Workflow is waiting for user input
    st.info("🤔 The agents are waiting for your input to continue the workflow...")
    user_response = st.chat_input("Continue the conversation...", key="workflow_input")
    
    if user_response:
        # Add user response to messages
        st.session_state.messages.append({
            "role": "user",
            "content": user_response,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Send response to workflow via queue
        st.session_state.message_queue.put({
            "type": "user_input",
            "content": user_response
        })
        
        st.rerun()
        
elif not st.session_state.processing:
    # Normal input for starting new workflow
    user_input = st.chat_input("Ask a question about your data...")
    
    if user_input:
        if not data_paths:
            st.error("Please upload at least one dataset first!")
        else:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Set processing state
            st.session_state.processing = True
            st.session_state.workflow_complete = False
            
            # Add status message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Initializing multi-agent workflow...",
                "is_status": True,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Start workflow in background thread
            thread = threading.Thread(
                target=run_workflow_background,
                args=(data_paths, user_input, max_rounds, st.session_state.message_queue),
                daemon=True
            )
            thread.start()
            st.session_state.workflow_thread = thread
            
            st.rerun()
else:
    # Workflow is processing - show active input for potential user interaction
    user_input_during_workflow = st.chat_input("Type here if agents need your input...", key="workflow_active_input")
    
    if user_input_during_workflow:
        # Add user response to messages
        st.session_state.messages.append({
            "role": "user",
            "content": user_input_during_workflow,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Send response to workflow via queue
        st.session_state.message_queue.put({
            "type": "user_input",
            "content": user_input_during_workflow
        })
        
        st.rerun()

# Auto-refresh while processing and check for new messages
if st.session_state.processing:
    # Process messages from queue
    messages_added = False
    try:
        while True:
            message = st.session_state.message_queue.get_nowait()
            if message.get("type") == "workflow_complete":
                st.session_state.workflow_complete = True
                st.session_state.processing = False
            elif message.get("type") == "user_input_needed":
                # Don't add this to messages, just trigger UI update
                messages_added = True
            else:
                st.session_state.messages.append(message)
                messages_added = True
    except queue.Empty:
        pass
    
    # Refresh if new messages or just to update UI
    if messages_added:
        st.rerun()
    else:
        time.sleep(1)  # Check every second
        st.rerun()