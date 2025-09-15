# Multi-Agent Collaboration for Automated Data Science Workflows

A sophisticated LangGraph-powered multi-agent system that orchestrates specialized AI agents to handle complex data science workflows. The system features intelligent task decomposition, autonomous agent collaboration, and comprehensive data analysis capabilities with a modern Streamlit web interface.

## 🚀 Key Features

### Advanced Multi-Agent Architecture
- **LangGraph Workflow Engine**: State-based agent orchestration with intelligent task routing
- **Specialized Agent Roles**: Project Manager, Data Analyst, ML Engineer, and Business Translator
- **CodeAct Framework**: Autonomous code generation, execution, and result interpretation
- **Dynamic State Management**: Persistent workflow state with memory and context preservation

### Comprehensive Data Science Capabilities
- **Automated EDA**: Complete exploratory data analysis with visualizations and insights
- **ML Pipeline Development**: End-to-end machine learning workflow automation  
- **Document Intelligence**: PDF analysis and content extraction with advanced NLP
- **Web Research Integration**: Real-time information retrieval and synthesis
- **Code Execution Environment**: Safe Python code generation and execution

### Enterprise-Ready Features
- **Interactive Web Interface**: Professional Streamlit-based dashboard
- **Workflow Visualization**: Real-time agent collaboration and state tracking
- **File Upload Support**: Handle CSV datasets and PDF documents
- **Conversation Memory**: Persistent chat history and context management
- **Configurable Agents**: YAML-based prompt templates and agent customization

## Architecture Overview

This is a **multi-agent data science workflow system** built with **LangGraph** and **CrewAI**. The system uses specialized AI agents to collaborate on complex data science tasks through an interactive Streamlit web interface.

### Core Components

**Multi-Agent Framework Structure:**
- `workflow/` - LangGraph-based workflow orchestration
  - `graph.py` - Main workflow graph definition
  - `nodes.py` - Individual workflow node implementations  
  - `state.py` - Shared state management across agents
- `crew/` - CrewAI agent implementations
  - `agents/` - Specialized agent definitions and configurations
  - `crew_tools.py` - Shared tools for agent operations
- `main.py` - Streamlit web application entry point

**Agent Types:**
- **Project Manager** - Task planning and coordination
- **Data Analyst** - Exploratory data analysis and insights
- **ML Engineer** - Machine learning model development
- **Business Translator** - Converting technical results to business insights

**Tool System:**
- `tools/sandbox_manager.py` - Safe code execution environment
- `crew/crew_tools.py` - Core agent tools (dataset loading, code execution, file operations)
- `utils/file_handler.py` - File upload and management utilities

## Development Commands

**Package Management:**
- `uv sync` - Install/sync dependencies (uses uv package manager)
- `uv add <package>` - Add new dependency
- `uv remove <package>` - Remove dependency

**Running the Application:**
- `streamlit run main.py` - Start the web interface (runs on http://localhost:8501)

**Testing:**
- `python test_workflow.py` - Test workflow functionality
- `python test_agent_tools.py` - Test agent tools and dataset detection
- `pytest` - Run full test suite (if available)

**Code Quality:**
- `black .` - Format code
- `ruff check .` - Lint code
- `ruff check . --fix` - Auto-fix linting issues
- `mypy .` - Type checking

## Key Technical Details

**State Management:**
- Uses LangGraph `StateGraph` for workflow orchestration
- State is passed between agents as a shared data structure
- Persistent state management through the workflow execution

**Code Execution:**
- Agents can execute Python code through the sandbox manager
- Code results are captured and shared between agents
- Execution history is maintained in the workflow state

**Data Handling:**
- CSV datasets uploaded to `workspace/` directory
- Automatic dataset detection and loading
- Pandas-based data manipulation throughout the pipeline

**Agent Communication:**
- Agents communicate through the workflow state
- Task results and intermediate outputs are preserved
- Memory and context maintained across agent handoffs

## Environment Setup

**Required Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key for agent LLMs

**Directory Structure:**
- `workspace/` - User uploaded datasets and generated files
- `.venv/` - Python virtual environment (created by uv)

## Important Notes

- This is a **Python 3.11+** project using modern async/await patterns
- The system uses **OpenAI GPT models** for agent intelligence
- Code execution happens in a controlled sandbox environment
- The Streamlit interface provides real-time workflow visualization
- Agents can autonomously generate, execute, and iterate on code solutions