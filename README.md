# auto-ds-agents

Multi-agent collaboration for automated data-science workflows, including a **Business Action Translator Agent** that turns model insights into actionable recommendations.

This repo is intentionally simple and friendly.  
Each agent lives in its own file, prompts live in YAML files, and one small orchestrator (`app.py`) strings things together.

---

## Project Structure (brief)

auto-ds-agents/
├─ .env
├─ .gitignore
├─ README.md
├─ requirements.txt
│
├─ prompts/ # one YAML per agent
│ ├─ data_cleaner.yaml
│ ├─ feature_engineer.yaml
│ ├─ model_selector.yaml
│ ├─ evaluator.yaml
│ └─ business_action_translator.yaml
│
├─ data/
│ ├─ raw/ # put raw CSVs here
│ └─ processed/ # save cleaned/feature data here
│
├─ src/
│ └─ auto_ds_agents/
│ ├─ init.py
│ ├─ app.py # orchestrator: runs agents in sequence
│ ├─ utils.py # small helpers (I/O, metrics, etc.)
│ └─ agents/
│ ├─ init.py
│ ├─ data_cleaner.py
│ ├─ feature_engineer.py
│ ├─ model_selector.py
│ ├─ evaluator.py
│ └─ business_action_translator.py
│
└─ tests/
└─ test_app.py
