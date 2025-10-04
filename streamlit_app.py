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
import re
import threading
import queue
import sys
from io import StringIO

load_dotenv()

def parse_agent_conversation(log_content):
    """Parse agent conversations from log content"""
    conversations = []
    lines = log_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for agent conversation pattern: "AgentName (to recipient):"
        agent_match = re.match(r'^(\w+) \(to ([^)]+)\):$', line)
        if agent_match:
            agent_name = agent_match.group(1)
            recipient = agent_match.group(2)
            
            # Collect the message content (next non-empty lines until separator or next agent)
            message_lines = []
            i += 1
            
            while i < len(lines):
                current_line = lines[i]
                
                # Stop if we hit a separator or next agent
                if (current_line.strip() == "--------------------------------------------------------------------------------" or
                    re.match(r'^\w+ \(to [^)]+\):$', current_line.strip()) or
                    current_line.strip().startswith("Next speaker:") or
                    current_line.strip().startswith("***** Suggested tool call")):
                    break
                
                if current_line.strip():  # Only add non-empty lines
                    message_lines.append(current_line)
                
                i += 1
            
            # Join the message content
            message_content = '\n'.join(message_lines).strip()
            
            if message_content:  
                conversations.append({
                    'agent': agent_name,
                    'recipient': recipient,
                    'content': message_content
                })
        else:
            i += 1
    
    return conversations

def get_agent_avatar(agent_name):
    """Get appropriate avatar for each agent"""
    avatars = {
        'User': '👤',
        'BusinessAnalyst': '📊',
        'DataExplorer': '🔍',
        'DataCleaner': '🧹',
        'FeatureEngineer': '⚙️',
        'Coder': '💻',
        'DataEngineer': '🏗️',
        'Modeler': '🤖',
        'Evaluator': '📈',
        'BusinessTranslator': '🔄'
    }
    return avatars.get(agent_name, '🤖')

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
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "processing" not in st.session_state:
    st.session_state.processing = False
if "agent_sequence" not in st.session_state:
    st.session_state.agent_sequence = []
if "user_response_needed" not in st.session_state:
    st.session_state.user_response_needed = False
if "user_response_prompt" not in st.session_state:
    st.session_state.user_response_prompt = ""
if "user_response" not in st.session_state:
    st.session_state.user_response = None
if "workflow_complete" not in st.session_state:
    st.session_state.workflow_complete = False
if "workflow_thread" not in st.session_state:
    st.session_state.workflow_thread = None
if "message_queue" not in st.session_state:
    st.session_state.message_queue = queue.Queue()
if "conversation_counter" not in st.session_state:
    st.session_state.conversation_counter = 0

# Header with native Streamlit styling
st.title("Minion Studio")
st.caption("Multi-Agent Data Science Workflows with Live Interaction")

# Sidebar
with st.sidebar:
    st.header("Data Configuration")
    
    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Upload Dataset(s)",
        type=["csv", "xlsx", "json"],
        accept_multiple_files=True,
        help="Upload one or more datasets for analysis"
    )
    
    # Save uploaded files
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
            for file in uploaded_files:
                st.caption(f"  • {file.name}")
    
    st.divider()
    
    # Workflow Configuration
    st.header("Workflow Settings")
    
    max_rounds = st.slider(
        "Max Conversation Rounds",
        min_value=10,
        max_value=100,
        value=50,
        step=10
    )
    
    human_input_mode = st.selectbox(
        "Human Input Mode",
        ["NEVER", "TERMINATE", "ALWAYS"],
        index=0,
        help="NEVER: Fully autonomous, TERMINATE: Ask at key points, ALWAYS: Interactive mode"
    )
    
    st.divider()
    
    # Agent Status Panel
    st.header("Agent Pipeline")
    
    agents_info = [
        ("Business Analyst", "Understand requirements"),
        ("Data Explorer", "Explore dataset"),
        ("Data Cleaner", "Clean data"),
        ("Feature Engineer", "Create features"),
        ("Modeler", "Build models"),
        ("Business Translator", "Explain results")
    ]
    
    for agent_name, agent_desc in agents_info:
        if st.session_state.current_agent == agent_name:
            st.success(f"**{agent_name}** _(Active)_")
            st.caption(f"   {agent_desc}")
        elif agent_name in st.session_state.agent_sequence:
            st.info(f"~~{agent_name}~~ (Complete)")
        else:
            st.text(f"{agent_name}")
            st.caption(f"   {agent_desc}")
    
    st.divider()
    
    # Clear button
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_agent = None
        st.session_state.agent_sequence = []
        st.session_state.processing = False
        st.session_state.user_response_needed = False
        st.session_state.workflow_complete = False
        st.rerun()

# Main content area
col1, col2 = st.columns([7, 3])

with col1:
    # Chat container
    st.subheader("Conversation")
    
    chat_container = st.container(height=500)
    
    with chat_container:
        # Display messages with live updating
        for message in st.session_state.messages:
            if message.get("is_status"):
                st.info(f"{message['content']}")
            elif message.get("is_next_speaker"):
                # Next speaker announcement
                st.markdown(f"**{message['content']}**")
                st.markdown("")  # Empty line
            elif message["role"] == "separator":
                # Separator line
                st.markdown("---")
                st.markdown("")  # Empty line
            elif message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
                    if message.get("timestamp"):
                        st.caption(f"[{message['timestamp']}]")
            elif message["role"] == "assistant":
                # Agent message
                agent_name = message.get("sender", "Assistant")
                recipient = message.get("recipient")
                avatar = message.get("avatar", "🤖")
                is_live = message.get("is_live", False)
                
                with st.chat_message("assistant", avatar=avatar):
                    # Show agent conversation format
                    if agent_name != "Assistant" and recipient:
                        st.markdown(f"**{agent_name} (to {recipient}):**")
                    elif agent_name != "Assistant":
                        st.markdown(f"**{agent_name}**")
                    
                    st.markdown("")  # Empty line like in conversation.txt
                    st.markdown(message["content"])
                    
                    if message.get("timestamp"):
                        st.caption(f"[{message['timestamp']}]")
        
        # Auto-scroll to bottom for live conversations
        if st.session_state.processing and st.session_state.messages:
            st.empty()  # This helps trigger a scroll to bottom

with col2:
    # Status panel
    st.subheader("Workflow Status")
    
    # Live conversation counter for real-time updates
    if st.session_state.processing:
        conversation_count = st.session_state.get("conversation_counter", 0)
        st.metric("Live Conversations", conversation_count)
        
        # Auto-refresh during processing
        if conversation_count > 0:
            time.sleep(0.1)  # Small delay to prevent too rapid updates
            st.rerun()
    
    if st.session_state.processing:
        st.warning(f"Processing with {st.session_state.current_agent or 'Initializing'}...")
        progress = len(st.session_state.agent_sequence) / len(agents_info)
        st.progress(progress)
        st.caption(f"Progress: {len(st.session_state.agent_sequence)}/{len(agents_info)} agents")
    elif st.session_state.workflow_complete:
        st.success("Workflow Complete!")
        st.metric("Agents Executed", len(st.session_state.agent_sequence))
    else:
        st.info("Ready to start")
    
    # Show current agent's task
    if st.session_state.current_agent:
        st.write("**Current Task:**")
        for name, desc in agents_info:
            if name == st.session_state.current_agent:
                st.caption(f"{desc}")
                break
    
    # Timestamp
    if st.session_state.messages:
        last_msg = st.session_state.messages[-1]
        if last_msg.get("timestamp"):
            st.caption(f"Last update: {last_msg['timestamp']}")

# Example prompts section
if not st.session_state.processing:
    st.divider()
    st.subheader("Example Prompts")
    example_cols = st.columns(4)
    
    examples = [
        "Find best ROI properties",
        "Predict house prices",
        "Property segmentation",
        "Investment analysis"
    ]
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Auto-fill the chat input (we'll handle this below)
                st.session_state.pending_input = example
                st.rerun()

# Input area
st.divider()

# Check if we have a pending example input
pending_input = st.session_state.get("pending_input", None)
if pending_input:
    del st.session_state.pending_input
    user_input = pending_input
else:
    # Handle user response during workflow
    if st.session_state.user_response_needed:
        st.warning(f"Agent is waiting: {st.session_state.user_response_prompt}")
        st.info("The workflow is paused until you provide a response.")
        
        user_response = st.chat_input(
            "Enter your response to continue the workflow...",
            key="user_response_input"
        )
        
        if user_response:
            st.session_state.user_response = user_response
            st.session_state.user_response_needed = False
            st.session_state.messages.append({
                "role": "user",
                "content": user_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.success("Response received! Continuing workflow...")
            # Don't rerun here, let the workflow continue naturally
        user_input = None
    else:
        user_input = st.chat_input(
            "Ask a question about your data...",
            key="chat_input"
        )

# Process new user input
if user_input and not st.session_state.processing:
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
        st.session_state.agent_sequence = []
        
        # Add status message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Initializing multi-agent workflow...",
            "is_status": True,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        st.rerun()

# Function to run workflow in background thread
def run_workflow_in_background():
    """Run the workflow in a background thread"""
    # Get the data paths and user query from session state
    data_paths = st.session_state.get('workflow_data_paths', [])
    user_query = st.session_state.get('workflow_user_query', '')
    max_rounds = st.session_state.get('workflow_max_rounds', 50)
    
    try:
        # Custom stdout capture to intercept agent messages
        class ConversationCapture:
            def __init__(self, original_stdout):
                self.original_stdout = original_stdout
                self.buffer = ""
                self.current_agent = None
                self.current_recipient = None
                self.current_content = []
                
            def write(self, text):
                # Write to original stdout for debugging
                if self.original_stdout:
                    self.original_stdout.write(text)
                
                if text:
                    self.buffer += text
                    # Process complete lines
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:
                            if line.strip():
                                self.process_line(line.strip())
                        self.buffer = lines[-1]
                return len(text)
                
            def process_line(self, line):
                # Parse conversation patterns and send to queue
                if line.startswith("Next speaker:"):
                    speaker = line.replace("Next speaker:", "").strip()
                    if speaker != "_Group_Tool_Executor":
                        # Add message directly to session state
                        st.session_state.messages.append({
                            "role": "system",
                            "content": f"Next speaker: {speaker}",
                            "is_next_speaker": True,
                            "timestamp": datetime.now().strftime("%H:%M:%S")
                        })
                        st.session_state.conversation_counter += 1
                        
                elif " (to " in line and "):" in line and not line.startswith("*****"):
                    try:
                        agent = line.split(" (to ")[0]
                        recipient = line.split(" (to ")[1].split("):")[0]
                        if agent != "_Group_Tool_Executor":
                            # Save current conversation if exists
                            if self.current_agent and self.current_content:
                                content = '\n'.join(self.current_content).strip()
                                if content:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "sender": self.current_agent,
                                        "recipient": self.current_recipient,
                                        "content": content,
                                        "avatar": get_agent_avatar(self.current_agent),
                                        "timestamp": datetime.now().strftime("%H:%M:%S")
                                    })
                                    st.session_state.conversation_counter += 1
                            
                            # Start new conversation
                            self.current_agent = agent
                            self.current_recipient = recipient
                            self.current_content = []
                    except:
                        pass
                        
                elif line == "--------------------------------------------------------------------------------":
                    self.queue.put({"type": "separator"})
                    
                else:
                    # Regular content line
                    if not any(line.startswith(x) for x in ["*****", ">>>>", "Call ID:", "Arguments:", "Input arguments:", "Output:"]):
                        self.queue.put({
                            "type": "content",
                            "text": line
                        })
                        
            def flush(self):
                if self.original_stdout:
                    self.original_stdout.flush()
        
        # Custom UserProxyAgent for Streamlit integration
        class StreamlitUserProxy(UserProxyAgent):
            def get_human_input(self, prompt: str = "") -> str:
                """Get input from Streamlit UI - pauses workflow until user responds"""
                st.session_state.user_response_needed = True
                st.session_state.user_response_prompt = prompt or "Please provide your input:"
                st.session_state.user_response = None
                
                # Add a message indicating we're waiting for user
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Waiting for user input: {prompt}",
                    "is_status": True,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Update conversation counter to trigger UI refresh
                st.session_state.conversation_counter = st.session_state.get("conversation_counter", 0) + 1
                
                # Return empty string to continue for now (Streamlit will handle this differently)
                return ""
        
        # Initialize agents
        ba = BusinessAnalyst()
        
        # Setup code execution config
        code_exec_config = {
            "work_dir": tempfile.mkdtemp(),
            "use_docker": False,
            "timeout": 120,
            "last_n_messages": 3
        }
        
        # Use NEVER mode for smoother workflow, handle user interaction through messages
        user = UserProxyAgent(
            name="User",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=code_exec_config,
            system_message="You are a user providing feedback and answering questions."
        )
        
        data_explorer = DataExplorer()
        data_cleaner = DataCleaner()
        feature_engineer = FeatureEngineer()
        coder = Coder()
        data_engineer = DataEngineer()
        modeler = Modeler()
        evaluator = Evaluator()
        business_translator = BusinessTranslator()
        
        # Agent list for tracking
        agent_list = [
            ("Business Analyst", ba),
            ("Data Explorer", data_explorer),
            ("Data Cleaner", data_cleaner),
            ("Feature Engineer", feature_engineer),
            ("Modeler", modeler),
            ("Business Translator", business_translator)
        ]
        
        # Create a custom group chat manager to capture conversations
        class ConversationCapture:
            def __init__(self):
                self.messages_captured = []
                
            def capture_message(self, sender_name, recipient_name, content):
                # Add "Next speaker" announcement
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"Next speaker: {sender_name}",
                    "is_next_speaker": True,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Add the agent conversation
                st.session_state.messages.append({
                    "role": "assistant",
                    "sender": sender_name,
                    "recipient": recipient_name,
                    "content": content,
                    "avatar": get_agent_avatar(sender_name),
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "is_live": True
                })
                
                # Add separator
                st.session_state.messages.append({
                    "role": "separator",
                    "content": "--------------------------------------------------------------------------------",
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
                # Force UI refresh
                if "conversation_counter" not in st.session_state:
                    st.session_state.conversation_counter = 0
                st.session_state.conversation_counter += 1
        
        conversation_capture = ConversationCapture()
        
        # Custom reply function to capture messages
        def custom_reply_func(recipient, messages, sender, config):
            """Capture agent messages in conversation.txt format"""
            try:
                if hasattr(sender, 'name') and messages:
                    agent_name = sender.name
                    # Find the display name
                    for display_name, agent_obj in agent_list:
                        if agent_obj == sender:
                            agent_name = display_name
                            st.session_state.current_agent = display_name
                            if display_name not in st.session_state.agent_sequence:
                                st.session_state.agent_sequence.append(display_name)
                            break
                    
                    # Extract the last message content
                    if isinstance(messages, list) and len(messages) > 0:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict) and 'content' in last_msg:
                            content = last_msg['content']
                            
                            # Get recipient name
                            recipient_name = "chat_manager"
                            if hasattr(recipient, 'name'):
                                recipient_name = recipient.name
                            
                            # Directly add the conversation to session state
                            # Add next speaker announcement
                            st.session_state.messages.append({
                                "role": "system",
                                "content": f"Next speaker: {agent_name}",
                                "is_next_speaker": True,
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Add the conversation
                            st.session_state.messages.append({
                                "role": "assistant",
                                "sender": agent_name,
                                "recipient": recipient_name,
                                "content": content,
                                "avatar": get_agent_avatar(agent_name),
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Add separator
                            st.session_state.messages.append({
                                "role": "separator",
                                "content": "--------------------------------------------------------------------------------",
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Update counter for UI refresh
                            st.session_state.conversation_counter = st.session_state.get("conversation_counter", 0) + 1
            except Exception as e:
                # Debug any errors
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"[DEBUG ERROR] {str(e)}",
                    "is_status": True,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                        
            return False, None
        
        # Register reply functions for all agents
        for display_name, agent in agent_list:
            agent.register_reply([UserProxyAgent, None], custom_reply_func, position=0)
        
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
        
        # Get user's last message
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        if user_messages:
            user_query = user_messages[-1]["content"]
            
            # Prepare message with dataset paths
            if len(data_paths) == 1:
                full_message = f"Dataset path: {data_paths[0]}\n{user_query}"
            else:
                paths_str = "\n".join([f"Dataset {i+1}: {path}" for i, path in enumerate(data_paths)])
                full_message = f"Dataset paths:\n{paths_str}\n\n{user_query}"
            
            # Run the workflow
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Starting agent collaboration...",
                "is_status": True,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Debug: Add a test message to see if UI is updating
            st.session_state.messages.append({
                "role": "system",
                "content": "[DEBUG] Starting group chat execution...",
                "is_status": True,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # Simple approach: Just run the workflow
            # The custom_reply_func should capture messages in real-time
            try:
                result, final_context, last_agent = initiate_group_chat(
                    pattern=pattern,
                    messages=full_message,
                    max_rounds=max_rounds
                )
                
                # Debug: Check if result has any messages
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"[DEBUG] Workflow completed. Result type: {type(result)}",
                    "is_status": True,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "system",
                    "content": f"[ERROR] {str(e)}",
                    "is_status": True,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })
            
            # Mark as complete
            st.session_state.workflow_complete = True
            st.session_state.processing = False
            
            # Add completion message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Analysis complete! The agents have finished processing your request.",
                "is_status": True,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            st.rerun()
        
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error occurred: {str(e)}",
            "is_status": True,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        st.session_state.processing = False
        st.session_state.workflow_complete = False
        st.rerun()

# Auto-refresh periodically when processing
if st.session_state.processing and not st.session_state.workflow_complete:
    time.sleep(2)
    st.rerun()