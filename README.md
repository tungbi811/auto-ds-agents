auto-ds-agents

Multi-agent collaboration for automated data-science workflows, including a dedicated Business Action Translator agent that turns model insights into actionable recommendations.

This repo is intentionally simple and friendly. Each agent lives in its own file, prompts live in YAML files, and one small orchestrator (app.py) strings things together.

Project Structure (brief)
auto-ds-agents/
├─ .env
├─ .gitignore
├─ README.md
├─ requirements.txt
│
├─ prompts/
│  ├─ data_cleaner.yaml
│  ├─ feature_engineer.yaml
│  ├─ model_selector.yaml
│  ├─ evaluator.yaml
│  └─ business_action_translator.yaml
│
├─ data/
│  ├─ raw/
│  └─ processed/
│
├─ src/
│  └─ auto_ds_agents/
│     ├─ __init__.py
│     ├─ app.py                 # orchestrator: runs agents in sequence
│     ├─ utils.py               # small helpers (I/O, metrics, etc.)
│     └─ agents/
│        ├─ __init__.py
│        ├─ data_cleaner.py
│        ├─ feature_engineer.py
│        ├─ model_selector.py
│        ├─ evaluator.py
│        └─ business_action_translator.py
│
└─ tests/
   └─ test_app.py


Prompts: one YAML per agent in prompts/.

Data: put raw CSVs in data/raw/; save cleaned/feature data in data/processed/.

Runs/outputs: you can add a runs/ folder later if you want to save logs/reports.
