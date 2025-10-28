# 🤓 Multi-Agent System for Automated Data Science Workflows

A modular **multi-agent system (MAS)** that autonomously executes the entire data science workflow — from business problem understanding to model execution and actionable recommendation generation.  
Built using **AG2** and **GPT-4o-mini**, the system coordinates multiple specialized agents that collaborate asynchronously to transform business questions into data-driven insights.

---

## 🚀 Overview

Modern data science workflows involve complex, multi-stage tasks — from data preparation to modeling and interpretation.  
This project implements a **multi-agent architecture** where each agent specializes in a specific stage, enabling modularity, explainability, and automation.

### 🔹 Agents
| Agent | Role | Model | Temperature |
|--------|------|--------|--------------|
| **Business Analyst** | Interprets business requirements and structures objectives | GPT-4o-mini | 0.5 |
| **Business Translator** | Converts analytical results into actionable business recommendations | GPT-4o-mini | 0.3 |
| **Data Scientist** | Designs the analytical plan and selects ML techniques | GPT-4o-mini | 0.3 |
| **Coder** | Generates and executes Python code for data analysis and modeling | GPT-4o-mini | 0.0 |

---

## ⚙️ System Architecture

```
AUTO-DS-AGENTS/
├── artifacts/                   # Generated models, reports, and visualizations
├── configs/                     # Configuration files (e.g., environment, prompts)
├── data/                        # Input datasets (CSV or structured files)
├── logs/                        # Agent conversation and execution logs
├── multi_agents/                # Core multi-agent system
│   ├── __init__.py
│   ├── business_analyst.py      # Interprets business problems and objectives
│   ├── business_translator.py   # Converts results into actionable recommendations
│   ├── coder.py                 # Generates and executes Python code
│   ├── data_scientist.py        # Designs analytical strategy and methods
│   └── group_chat.py            # Coordinates communication among agents
├── utils/                       # Helper modules and shared utilities
│   ├── sidebar.py               # (Optional) Streamlit/CLI integration support
│   └── utils.py                 # Common helper functions
├── main.py                      # Main entry point for executing a workflow
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Ignored files and folders
└── README.md                    # Project documentation

```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- OpenAI API key
- Basic Python data-science libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Setup Steps

```bash
git clone https://github.com/tungbi811/Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows.git
cd Multi-Agent-Collaboration-for-Automated-Data-Science-Workflows

# Create virtual environment
uv venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API key
OPENAI_API_KEY=your_api_key_here
```

---

## 🎯 Usage

Run the main system to start agent collaboration:

```bash
python main.py
```

Each run processes **one dataset and one analytical query** at a time.

### Example Input (User Agent)

> **Question:** How can we accurately estimate the market value of a house given its features?
>
> **Dataset:** house_prices.csv

The agents then collaborate as follows:
1. **Business Analyst** extracts objectives and defines the ML problem type (regression).
2. **Business Translator** formulates clear analytical tasks for the Data Scientist.
3. **Data Scientist** designs the modeling strategy.
4. **Coder** executes the code via `JupyterCodeExecutor` and returns results.
5. **Business Translator** produces the final business recommendations in Markdown.

---

## 🔬 Example Output

```
## Business Recommendations
- **Insight:** Location and overall quality have the strongest influence on house prices.
- **Recommendation:** Focus on improving property quality in mid-range neighborhoods to maximize value.
- **Next Step:** Consider developing price prediction dashboards for real-time valuation updates.
```

---

## 🛡️ Safety & Design Principles

- Agents communicate asynchronously using AG2 message routing.
- `JupyterCodeExecutor` ensures code runs safely in a sandboxed backend environment.
- Each agent follows structured input/output schemas (Pydantic models) for consistency.
- The system maintains full execution logs for transparency and traceability.

---

## 🕈️ Future Enhancements

- [ ] Add benchmarking and performance metrics for generated models
- [ ] Expand to multi-query workflows
- [ ] Integrate data visualization dashboards
- [ ] Support for multi-model agents (e.g., Claude, Gemini)

---

**Developed by:** Monika Shakya, Van Thang Doan, Yamuna G C, Linh Chi Tong, Szu-Yu Lin, Duy Tung Nguyen  
**License:** MIT  
**Built with ❤️ for data science automation and intelligent workflows**
