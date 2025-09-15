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

## 🏗️ System Architecture

```
├── main.py                    # Streamlit web application & workflow orchestration
├── agents/                    # Multi-agent system core
│   ├── workflow.py           # LangGraph workflow definition & state management
│   ├── base_agent.py         # CodeAct agent base class
│   ├── project_manager.py    # Task planning & coordination agent
│   ├── data_analyst.py       # Data analysis & EDA specialist
│   ├── ml_engineer.py        # ML model development agent
│   └── business_translator.py # Business insights & communication agent
├── tools/                     # Specialized tool implementations
│   ├── basic_eda.py          # Automated exploratory data analysis
│   ├── web_search.py         # Web search & information retrieval
│   ├── document_analyze.py   # PDF processing & content extraction
│   └── code_execute.py       # Safe Python code execution environment
├── core/                      # System infrastructure
│   ├── state.py              # Workflow state management
│   └── memory.py             # Conversation & context memory
├── utils/                     # Shared utilities & helpers
│   ├── utils.py              # AI client & common functions
│   └── parsers.py            # Data parsing & validation utilities
├── prompts/                   # Agent configuration & templates
│   └── codeact_prompts.py    # CodeAct framework prompt templates
└── test/                      # Comprehensive testing suite
    ├── test_agents.py        # Agent functionality tests
    ├── test_workflow.py      # Workflow integration tests
    └── test_main.py          # Application end-to-end tests
```

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.11+**
- **OpenAI API Key** (GPT-4 recommended for optimal performance)
- **8GB+ RAM** (for ML workloads)

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tungbi811/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows.git
   cd Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows
   ```

2. **Environment Setup**
   ```bash
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configuration**
   ```bash
   cp .env.example .env
   # Edit .env file and add your OpenAI API key:
   echo "OPENAI_API_KEY=your_api_key_here" >> .env
   ```

4. **Launch Application**
   ```bash
   streamlit run main.py
   ```
   
   Access the application at: **http://localhost:8501**

## 🎯 Usage Guide

### Multi-Agent Workflow Capabilities

**Data Science Project Analysis**
- Upload CSV dataset → "Perform comprehensive analysis of this dataset"
- "Generate ML model recommendations for customer segmentation"  
- "Create a complete data science workflow for this business problem"

**Document Intelligence & Research**
- Upload PDF reports → "Extract key insights and create executive summary"
- "Research latest developments in machine learning and summarize findings"
- "Analyze this research paper and identify implementation opportunities"

**Advanced Code Development**
- "Build a complete ML pipeline for fraud detection"
- "Create automated data quality assessment tools"
- "Develop interactive dashboard for business metrics"

**Business Intelligence**
- "Translate technical findings into business recommendations"
- "Generate actionable insights from data analysis results" 
- "Create stakeholder presentation from ML model results"

### Agent Collaboration Examples

The system automatically coordinates multiple agents:

1. **Project Manager** → Plans and decomposes complex tasks
2. **Data Analyst** → Performs EDA and statistical analysis  
3. **ML Engineer** → Develops and evaluates ML models
4. **Business Translator** → Creates business-focused insights

## 🔧 Advanced Configuration

### Custom Agent Behavior

Edit `prompts/codeact_prompts.py` to modify agent personalities and capabilities:

```python
PROJECT_MANAGER_PROMPT = """
Your custom project manager instructions...
Focus areas: planning, coordination, timeline management
"""

DATA_ANALYST_PROMPT = """  
Your custom data analyst instructions...
Expertise: EDA, statistical analysis, data visualization
"""
```

### Workflow Customization

Modify `agents/workflow.py` to add new agent types or change collaboration patterns:

```python
def create_custom_workflow():
    # Add new agents or modify existing workflow
    workflow.add_node("custom_agent", custom_agent_function)
    workflow.add_edge("start", "custom_agent")
```

### Tool Integration

Add new capabilities by extending `tools/` directory:

```python
# tools/your_custom_tool.py
def your_custom_function(input_data):
    """Your custom tool implementation"""
    return processed_result
```

## 🛡️ Security & Safety

- **Sandboxed Execution**: Code runs in controlled environment with restricted imports
- **Input Validation**: All user inputs sanitized and validated
- **API Security**: Environment-based credential management
- **Memory Management**: Automatic cleanup of sensitive data
- **Error Handling**: Comprehensive error catching and recovery

## 🧪 Testing & Quality Assurance

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test categories
python test_agents.py      # Agent functionality tests
python test_workflow.py    # Workflow integration tests  
python test_main.py        # End-to-end application tests
```

### Test Coverage
- **Agent Behavior Testing**: Individual agent capabilities and responses
- **Workflow Integration**: Multi-agent collaboration scenarios
- **Tool Functionality**: All integrated tools and utilities
- **UI/UX Testing**: Streamlit interface and user interactions

## 🚀 Performance & Scalability

### System Requirements
- **Development**: 4GB RAM, Python 3.11+
- **Production**: 8GB+ RAM, GPU optional for large ML workloads
- **Enterprise**: 16GB+ RAM, distributed computing support

### Optimization Features
- **Caching**: Intelligent caching of API responses and computations
- **Async Processing**: Non-blocking operations for better UX
- **Memory Management**: Automatic cleanup and optimization
- **Batch Processing**: Efficient handling of large datasets

## 📊 Supported Data Formats & Integrations

### Data Sources
- **CSV Files**: Structured datasets for analysis
- **PDF Documents**: Research papers, reports, documentation
- **Web APIs**: Real-time data retrieval and research
- **JSON/XML**: Structured data import and processing

### Export Formats
- **Visualizations**: PNG, SVG, interactive HTML
- **Reports**: Markdown, PDF, HTML dashboards
- **Data**: CSV, JSON, Excel formats
- **Code**: Python scripts, Jupyter notebooks

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines:

### Development Workflow
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Follow coding standards (PEP 8, type hints, docstrings)
4. Add comprehensive tests for new features
5. Update documentation and README if needed
6. Submit Pull Request with detailed description

### Code Standards
- **Python Style**: PEP 8 compliance with Black formatting
- **Type Hints**: Full type annotation for better code quality
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit tests for all new functionality

## 📋 Technology Stack

### Core Framework
- **LangGraph**: Multi-agent workflow orchestration
- **Streamlit**: Modern web interface
- **OpenAI GPT-4**: Advanced language model capabilities
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms

### Supporting Libraries
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Document Processing**: PyPDF2, BeautifulSoup4
- **Web Integration**: Requests, HTTPX
- **State Management**: SQLAlchemy, SQLite
- **Testing**: Pytest, unittest

See `requirements.txt` for complete dependency list.

## 🎨 Roadmap & Future Enhancements

### Planned Features
- [ ] **Multi-Model Support**: Anthropic Claude, Google Gemini integration
- [ ] **Advanced Visualizations**: Interactive dashboards and reports  
- [ ] **Database Integration**: PostgreSQL, MongoDB connectors
- [ ] **Docker Deployment**: Containerized deployment options
- [ ] **API Endpoints**: RESTful API for programmatic access
- [ ] **Real-time Collaboration**: Multi-user workspace support

### Research & Development
- [ ] **Advanced ML Agents**: AutoML and hyperparameter optimization
- [ ] **Natural Language Interface**: Voice and conversational AI
- [ ] **Integration Platform**: Connect with popular data science tools
- [ ] **Enterprise Features**: User management, audit trails, compliance

## 👥 Development Team

**Core Contributors**
- **Linh Chi Tong** - Technical Lead & Architecture
- **Duy Tung Nguyen** - ML Engineering & Data Science
- **Monika Shakya** - Frontend & User Experience  
- **Van Thang Doan** - Backend & Infrastructure
- **Yamuna G C** - Testing & Quality Assurance
- **Szu-Yu Lin** - Documentation & DevOps

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.

## 🆘 Support & Community

### Getting Help
- **GitHub Issues**: [Report bugs and request features](https://github.com/tungbi811/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows/issues)
- **Documentation**: Comprehensive guides and API reference
- **Community Discussions**: Join our developer community

### Contributing
We're actively seeking contributors in:
- **ML Engineering**: Advanced algorithms and model optimization
- **Frontend Development**: UI/UX improvements and new features  
- **Documentation**: Technical writing and user guides
- **Testing**: Quality assurance and automated testing

---

**Built with ❤️ for the data science and AI community**

*Empowering data scientists with intelligent automation and collaborative AI*