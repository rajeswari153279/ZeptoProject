# Zepto Data & AI Platform — Capstone Project

An end-to-end AI/ML platform with three connected modules: a data engineering pipeline, an analytics + ML pipeline, and a GenAI support assistant.

## Modules
- `/data_pipeline` — Scrapes, cleans, and stores book data in SQLite; runs SQL queries (25 marks)
- `/analytics` — Titanic EDA, 3 ML classifiers, imbalance handling, hyperparameter tuning, regression (50 marks)
- `/support_assistant` — RAG-based GenAI assistant answering policy questions using LangGraph + ChromaDB (25 marks)

## Setup

Each module uses the same virtual environment approach:

```
python -m venv venv
.\venv\Scripts\Activate      # Windows
```

Each module has its own `requirements.txt`. Install per-module before running:
```
pip install -r data_pipeline/requirements.txt
pip install -r analytics/requirements.txt   # if present
pip install -r support_assistant/requirements.txt
```

## How to run each module

**Data Pipeline:**
```
python data_pipeline/scrape_and_build.py
```

**Analytics (run in order):**
```
python analytics/01_eda.py
python analytics/02_modeling.py
python analytics/03_imbalance_tuning.py
python analytics/04_regression_final.py
```

**Support Assistant:**
```
python support_assistant/a_build_embeddings.py
python support_assistant/b_langgraph_app.py
cd support_assistant
uvicorn main:app --reload
```
Then visit `http://127.0.0.1:8000/docs` to test the `/ask` endpoint.

**Docker (support assistant):**
```
cd support_assistant
docker build -t zepto-assistant .
docker run -p 8000:8000 zepto-assistant
```

## Design decisions summary

- **Data Pipeline:** Fixed GBP→INR rate of 105.50 (stated, not live-looked-up). Two-table SQLite schema (books, categories) with PK/FK for JOIN demonstration.
- **Analytics:** Stratified train/test split to preserve class balance. All preprocessing (imputation, scaling, encoding) fit only on training data to avoid leakage. Random Forest selected as final model based on F1-score comparison across 3 classifiers.
- **Support Assistant:** Runs in MOCK_LLM mode by default (no API key required, as required by the assignment). Uses `all-MiniLM-L6-v2` embeddings with ChromaDB for retrieval, and LangGraph to route between policy questions (retrieval-based) and general questions (direct response).
-
