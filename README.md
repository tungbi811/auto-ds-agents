# Auto-DS-Agents

**Auto-DS-Agents** is a modular, agent-based framework for automating end-to-end data science workflows.
It orchestrates specialized agents that handle different stages of the ML lifecycle—data cleaning, feature engineering, model selection, evaluation, and business action translation—making it easier to prototype and deploy automated data-driven solutions.

---

## 🚀 Features

* **Agent-Oriented Design**: Each task (e.g., cleaning, feature engineering, model selection) is handled by a dedicated agent.
* **Configurable Prompts**: YAML-based prompt files make it easy to customize agent behavior.
* **Data Pipeline Ready**: Built-in structure for raw and processed datasets.
* **Evaluation Support**: Automated evaluation to measure model performance.
* **Business Integration**: Converts results into business-oriented actions through a translator agent.
* **Test Coverage**: Includes unit tests for reliability.

---

## 📂 Project Structure

```
auto-ds-agents/
├── .env                     # Environment variables (API keys, secrets, etc.)
├── .gitignore
├── README.md
├── requirements.txt         # Python dependencies
│
├── prompts/                 # YAML prompts to guide agents
│   ├── data_cleaner.yaml
│   ├── feature_engineer.yaml
│   ├── model_selector.yaml
│   ├── evaluator.yaml
│   └── business_action_translator.yaml
│
├── data/                    # Data storage
│   ├── raw/                 # Unprocessed input data
│   └── processed/           # Cleaned/engineered datasets
│
├── src/
│   └── auto_ds_agents/
│       ├── __init__.py
│       ├── app.py           # Orchestrator: runs agents in sequence
│       ├── utils.py         # Helper functions (I/O, metrics, etc.)
│       └── agents/          # Individual agent implementations
│           ├── __init__.py
│           ├── data_cleaner.py
│           ├── feature_engineer.py
│           ├── model_selector.py
│           ├── evaluator.py
│           └── business_action_translator.py
│
└── tests/
    └── test_app.py          # Unit tests for the orchestrator
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/auto-ds-agents.git
cd auto-ds-agents
pip install -r requirements.txt
```

---

## ▶️ Usage

1. **Set up environment variables** in `.env` (e.g., API keys if using LLMs).
2. **Prepare data** inside the `data/raw/` directory.
3. **Run the orchestrator**:

```bash
python src/auto_ds_agents/app.py
```

4. **Processed outputs** will be saved in `data/processed/`.

---

## 🧪 Testing

Run tests with:

```bash
pytest tests/
```

---

## 🔧 Customization

* Modify YAML prompt files in `prompts/` to adjust agent behaviors.
* Extend agents by adding new modules under `src/auto_ds_agents/agents/`.
* Integrate with external APIs by updating `utils.py`.

---

## 📌 Roadmap

* [ ] Add visualization module for pipeline runs
* [ ] Support for distributed orchestration
* [ ] Additional business action translators for domain-specific use cases

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo, create a branch, and open a pull request.

---

## 📜 License

This project is licensed under the MIT License.
