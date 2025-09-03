import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from agents import agent_workflow

# Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize
load_dotenv()

@st.cache_resource
def get_openai_client():
    """Cache the OpenAI client to avoid recreating it"""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def display_visualizations(visualizations):
    """Helper function to display visualizations"""
    if visualizations:
        st.subheader("Data Visualizations")
        for title, plot_base64 in visualizations:
            if title != "Error":
                st.markdown(f"**{title}**")
                st.image(f"data:image/png;base64,{plot_base64}")
            else:
                st.error(plot_base64)

def main():
    st.title("Multi-Tool AI Assistant")
    st.markdown("Ask me anything! I can search the web, analyze documents, execute code, and analyze data.")
    
    # Get OpenAI client
    client = get_openai_client()
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("File Upload")
        uploaded_file = st.file_uploader(
            "Upload a file for analysis",
            type=['pdf', 'csv'],
            help="Upload a PDF for document analysis or CSV for data analysis"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Uploaded: {uploaded_file.name}")
        else:
            temp_file_path = None
        
        st.markdown("---")
        st.markdown("""
        **Available Tools:**
        * Web Search
        * PDF Analysis
        * Code Execution
        * Data Analysis
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
        # Display visualizations if they exist in the message
        if "visualizations" in message:
            display_visualizations(message["visualizations"])
    
    # Chat input
    if prompt := st.chat_input("What would you like me to help you with?"):
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Determine if file upload is needed
                    use_uploaded_file = uploaded_file is not None and any(
                        keyword in prompt.lower() 
                        for keyword in ["analyze", "data", "pdf", "document", "csv"]
                    )
                    
                    # Call agent workflow
                    result = agent_workflow(
                        prompt, 
                        client, 
                        file_path=temp_file_path if use_uploaded_file else None,
                        use_uploaded_file=use_uploaded_file
                    )
                    
                    response = result["response"]
                    search_used = result.get("search_used", False)
                    
                    # Add search indicator
                    if search_used:
                        st.info("🔍 Used web search for this response")
                    
                    st.markdown(response)

                    # Display visualizations if available
                    if "visualizations" in result:
                        display_visualizations(result["visualizations"])

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    # Save both response and visualizations:
                    assistant_message = {"role": "assistant", "content": response}
                    if "visualizations" in result and result["visualizations"]:
                        assistant_message["visualizations"] = result["visualizations"]
                    st.session_state.messages.append(assistant_message)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
    
    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# Monika Trail
if __name__ == "__main__":
    main()