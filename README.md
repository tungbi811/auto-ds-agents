# Multi-Agent Collaboration for Automated Data Science Workflows

A sophisticated multi-agent AI assistant that orchestrates specialized tools to handle diverse tasks including web search, document analysis, code execution, and data science workflows. Built with a modular architecture that intelligently routes user requests to the most appropriate agent based on intent detection.

## 🚀 Features

### Multi-Agent Architecture
- **Intelligent Intent Detection**: Automatically routes requests to specialized agents
- **Web Search Agent**: Real-time information retrieval and synthesis
- **Document Analysis Agent**: PDF processing and content extraction  
- **Code Execution Agent**: Safe Python code generation and execution
- **Data Analysis Agent**: Automated exploratory data analysis (EDA)

### Key Capabilities
- **Smart File Handling**: Upload and analyze PDFs and CSV files
- **Interactive Chat Interface**: Streamlit-based web UI with chat history
- **Configurable Prompts**: YAML-based prompt templates for easy customization
- **Visualization Support**: Automatic chart generation and display
- **Safety-First Code Execution**: Sandboxed environment with restricted imports

## 🏗️ Architecture

```
├── main.py              # Streamlit web application
├── agents/              # Core agent system
│   ├── base_agent.py    # Main workflow orchestrator
│   └── __init__.py      # Agent exports
├── tools/               # Specialized tool implementations
│   ├── web_search.py    # Web search functionality
│   ├── document_analyze.py  # PDF analysis tools
│   ├── code_execute.py  # Safe code execution
│   └── basic_eda.py     # Data analysis tools
├── utils/               # Shared utilities
│   └── utils.py         # Helper functions and AI client
└── prompts/             # Configuration files
    └── prompt.yaml      # Agent prompt templates
```

## ⚙️ Installation

### Prerequisites
- Python 3.11+
- OpenAI API key

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows.git
   cd Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key:
   OPENAI_API_KEY=your_api_key_here
   ```

## 🎯 Usage

### Start the Application
```bash
streamlit run main.py
```

The web interface will open at `http://localhost:8501`

### Supported Query Types

**Web Search Queries**
- "Search for the latest AI developments"
- "Find current weather in Tokyo"
- "What are the recent news about cryptocurrency?"

**Code Execution**
- "Calculate the factorial of 10"
- "Plot a sine wave"
- "Generate 100 random numbers and show statistics"

**Document Analysis**
- Upload a PDF and ask: "Summarize this document"
- "Extract key findings from the uploaded report"

**Data Analysis** 
- Upload a CSV file and ask: "Analyze this dataset"
- "Show me the correlation matrix for this data"
- "Generate visualizations for the uploaded data"

## 🔧 Configuration

### Customizing Agent Behavior

Edit `prompts/prompt.yaml` to modify how agents respond:

```yaml
web_search:
  template: |
    Your custom prompt template here...
  model: "gpt-3.5-turbo"
  temperature: 0.7

code_generation:
  template: |
    Your custom code generation prompt...
  model: "gpt-3.5-turbo" 
  temperature: 0.1
```

### Adding New Tools

1. Create a new tool in `tools/your_tool.py`
2. Add the handler function following the existing pattern
3. Update `agents/base_agent.py` to include your new intent detection
4. Add corresponding prompts in `prompts/prompt.yaml`

## 🛡️ Security Features

- **Sandboxed Code Execution**: Limited to approved Python libraries
- **No File System Access**: Code execution restricted from system operations
- **Input Validation**: All user inputs are sanitized and validated
- **API Key Protection**: Environment variables for sensitive credentials

## 🧪 Testing

The project includes comprehensive testing for all components:

```bash
# Run tests (when available)
python -m pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Ensure all agents maintain the same interface pattern

## 📋 Dependencies

### Core Dependencies
- **streamlit**: Web interface framework
- **openai**: OpenAI API client
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **beautifulsoup4**: Web scraping and HTML parsing
- **PyPDF2**: PDF processing
- **python-dotenv**: Environment variable management

See `requirements.txt` for the complete list of dependencies.

## 🎨 Future Enhancements

- [ ] Additional agent types (image analysis, audio processing)
- [ ] Multi-model support (Anthropic Claude, Google Gemini)
- [ ] Advanced data visualization options
- [ ] Conversation memory and context persistence
- [ ] Plugin system for custom agents
- [ ] REST API endpoints
- [ ] Docker containerization

## 👥 Contributors

- Monika Shakya
- Van Thang Doan  
- Yamuna G C
- Linh Chi Tong
- Szu-Yu Lin
- Duy Tung Nguyen

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
1. Check the [Issues](https://github.com/your-username/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows/issues) page
2. Create a new issue with detailed description
3. Join our community discussions

---

**Built with ❤️ for the data science and AI community**