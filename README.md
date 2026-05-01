# Customer Communication Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10%2B-yellow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> **Portfolio project — synthetic data only. No real customer data is used anywhere in this repository.**

---

## Live Demo

🚀 **[Launch Dashboard](https://customer-communication-intelligence-scq8nrviczgurn2djzwaok.streamlit.app/)**

The live Streamlit dashboard includes all 4 pages:
- Overview — KPIs, engagement distribution, segment breakdown
- Anomaly Monitor — complaint spikes, campaign heatmap
- Case Explorer — filterable table, model scores, drill-down
- AI Summary — next-best-action card with risk rating

---

## What This Project Does

A portfolio-grade data and AI pipeline that simulates a **Customer Communication Intelligence Platform** for a fictional financial services company. The system ingests synthetic customer communication data, detects anomalies, predicts which customers need intervention, and generates AI-assisted next-best-action summaries — all exposed via a FastAPI backend and a Streamlit dashboard.

---

## Why This Project Matters

Customer communication teams in regulated industries face a common challenge: large volumes of outreach data, limited visibility into which campaigns or customers are at risk, and slow manual review processes. This project demonstrates how a lightweight Python stack — without expensive enterprise tooling — can answer key business questions in near real-time.

---

## Business Questions Answered

- Which campaigns or customer segments are underperforming?
- Which cohorts are at risk of low engagement or escalation?
- Which communication cases need intervention right now?
- What short next-best-action summary should a business user see?

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                          │
│   generate_data.py → ingest.py → validate.py → DuckDB  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               FEATURE + MODEL LAYER                     │
│   build_features.py → train_model.py → anomaly.py      │
│                        score.py                         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    GENAI LAYER                          │
│        prompts.py → summarizer.py (OpenAI / stub)      │
└────────────────────────┬────────────────────────────────┘
                         │
        ┌────────────────┴──────────────────┐
        │                                   │
┌───────▼──────────┐             ┌──────────▼──────────┐
│   FastAPI API    │             │ Streamlit Dashboard  │
│  /health         │             │  Overview            │
│  /score-customer │             │  Anomaly Monitor     │
│  /detect-anomaly │             │  Case Explorer       │
│  /generate-summary             │  AI Summary          │
└──────────────────┘             └─────────────────────┘
```

**Storage:** DuckDB (single file, zero server setup, analytical SQL)

---

## Repo Structure

```
customer-communication-intelligence/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config/
│   └── config.yaml               ← all thresholds and settings
├── assets/
│   ├── screenshots/streamlit/    ← dashboard screenshots
│   └── screenshots/powerbi/      ← local Power BI screenshots
├── data/
│   ├── raw/                      ← generated CSV (gitignored)
│   └── processed/                ← DuckDB + processed CSV (gitignored)
├── notebooks/                    ← optional EDA notebook
├── dashboard/
│   └── powerbi/                  ← local Power BI only (see note)
├── src/
│   ├── data/
│   │   ├── generate_data.py      ← synthetic data generator
│   │   ├── ingest.py             ← generate → validate → DuckDB pipeline
│   │   └── validate.py           ← schema + business rule validation
│   ├── features/
│   │   └── build_features.py     ← feature engineering (26 features)
│   ├── modeling/
│   │   ├── train_model.py        ← Random Forest training + evaluation
│   │   ├── score.py              ← single + batch scoring
│   │   └── anomaly.py            ← 3 anomaly detectors
│   ├── genai/
│   │   ├── summarizer.py         ← OpenAI API + stub fallback
│   │   └── prompts.py            ← prompt templates
│   ├── api/
│   │   ├── main.py               ← FastAPI app (4 endpoints)
│   │   └── schemas.py            ← Pydantic request/response models
│   └── utils/
│       └── db.py                 ← DuckDB connection helpers
├── app/
│   └── streamlit_app.py          ← 4-page Streamlit dashboard
└── tests/                        ← pytest test suite (158 tests)
```

---

## ⚠ Synthetic Data Notice

**This project uses entirely synthetic, randomly generated data. No real customer records, personal information, or confidential business data is used or referenced anywhere in this repository.**

The dataset is generated by `src/data/generate_data.py` using `numpy.random` with a fixed seed (42) for full reproducibility.

---

## Dataset Schema

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Synthetic unique customer ID |
| `segment` | string | Premium / Standard / Basic |
| `product_type` | string | Auto / Home / Life / Health |
| `channel` | string | Email / SMS / Phone / Direct Mail |
| `campaign_id` | string | CAMP_001 through CAMP_010 |
| `sent_date` | date | Last 90 days |
| `opened` | int (0/1) | Communication opened |
| `clicked` | int (0/1) | Link clicked |
| `response_flag` | int (0/1) | Customer responded |
| `complaint_flag` | int (0/1) | Complaint raised |
| `escalation_flag` | int (0/1) | Escalation raised |
| `engagement_score` | float | Composite score 0.0–1.0 |
| `sentiment_text` | string | positive / neutral / negative / mixed |
| `premium_bucket` | string | Low / Mid / High |
| `tenure_months` | int | Months as customer (1–120) |
| `days_since_last_contact` | int | Days since last outreach |
| `opt_out_flag` | int (0/1) | Customer opted out |
| `needs_intervention` | int (0/1) | **Target variable** |

**Dataset stats:** 5,000 rows · 34.7% intervention rate · 4.9% complaint rate

---

## Target Variable Definition

```
needs_intervention = 1  when ANY of:
  - engagement_score < 0.25  AND  opened == 0  AND  days_since_contact < 45
  - complaint_flag == 1
  - escalation_flag == 1
  - days_since_last_contact > 120  AND  response_flag == 0
    AND  opt_out_flag == 0  AND  engagement_score < 0.5
```

---

## Modeling Approach

**Algorithm:** Random Forest Classifier (scikit-learn)

**Feature engineering** (`build_features.py`):
- 9 passthrough numeric features
- 4 interaction features: `contact_but_no_response`, `high_risk_combo`, `engagement_x_tenure`, `low_engagement_long_silence`
- One-hot encoded categoricals: segment, product type, channel, premium bucket, sentiment

**26 total features** after encoding.

**Evaluation:** 5-fold cross-validation · ROC-AUC · Precision/Recall/F1

> Note: ROC-AUC is near 1.0 on this dataset because the target is derived deterministically from the same features. In production, expect noisier signals and lower AUC.

---

## Anomaly Detection

Three lightweight pandas-based detectors in `src/modeling/anomaly.py`:

| Detector | Method | Trigger |
|---|---|---|
| Segment Engagement Drop | Week-over-week mean engagement per segment | Drop > 15% |
| Complaint Spike | Z-score on daily complaint counts | Z-score > 2.0 |
| Campaign Underperformance | Campaign open rate vs median (data-only campaigns) | < 50% of median |

All thresholds are configurable in `config/config.yaml`.

---

## GenAI Summary Feature

`src/genai/summarizer.py` generates a 2–3 sentence next-best-action summary per customer case.

**Example output:**
> "This Basic segment customer is showing elevated intervention risk (score: 0.84) with negative sentiment and 147 days since last contact. Recommend escalating from SMS to a direct phone call to re-establish engagement and address any outstanding concerns. Broader segment-level anomalies have been detected — review campaign messaging for this cohort before next outreach."

**Two paths:**
- **OpenAI path:** Uses `gpt-3.5-turbo` when `OPENAI_API_KEY` is set
- **Stub path:** Template-filled summary using real data values — works offline, no cost

Set `OPENAI_API_KEY=` in `.env` to use the stub (default for demo).

---

## API Endpoints

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/score-customer` | Score a single customer case |
| `POST` | `/detect-anomaly` | Run anomaly detectors |
| `POST` | `/generate-summary` | Generate next-best-action summary |

Full interactive docs at: `http://localhost:8000/docs`

---

## Streamlit Dashboard Pages

| Page | What it shows |
|---|---|
| **Overview** | KPI cards, engagement distribution, segment breakdown |
| **Anomaly Monitor** | Anomaly flags, complaint spike chart, campaign heatmap |
| **Customer Case Explorer** | Filterable case table, drill-down, model scores |
| **AI Summary** | Select a case → generate next-best-action card |

---

## Power BI Local Reporting

Power BI is **optional and local-only**. See `dashboard/powerbi/README.md` for instructions on connecting to the processed CSV export and generating screenshots.

> Only screenshots (PNG/PDF) should be committed to the repo — never the `.pbix` file with embedded data.

---

## Screenshots

### Streamlit Dashboard

| Page | Screenshot |
|---|---|
| Overview | ![Overview](assets/screenshots/streamlit/overview.png) |
| Anomaly Monitor | ![Anomaly Monitor](assets/screenshots/streamlit/anomaly_monitor.png) |
| Customer Case Explorer | ![Case Explorer](assets/screenshots/streamlit/customer_case_explorer.png) |
| AI Summary | ![AI Summary](assets/screenshots/streamlit/ai_summary.png) |

> TODO: Add screenshots after Streamlit app is running locally.

### Power BI (local only)

![Power BI Overview](assets/screenshots/powerbi/powerbi_overview.png)

> TODO: Add after generating Power BI screenshots locally.

---

## Setup Instructions

### Prerequisites

- Python 3.11 or 3.12 (recommended — Python 3.14 may have package compatibility issues)
- Git

### 1. Clone the repo

```bash
git clone https://github.com/GenAIFinance/customer-communication-intelligence.git
cd customer-communication-intelligence
```

### 2. Install dependencies

```bash
pip install -r requirements.txt --prefer-binary
```

### 3. Create your .env file

```bash
# Linux / Mac
cp .env.example .env

# Windows CMD
copy .env.example .env
```

Edit `.env` and set:
```
OPENAI_API_KEY=        # leave blank to use stub fallback
DUCKDB_PATH=data/processed/communications.duckdb
```

---

## How to Run Locally

### Generate data + run full pipeline

```bash
python -m src.data.ingest
```

### Train the model

```bash
python -m src.modeling.train_model
```

### Run anomaly detection

```bash
python -m src.modeling.anomaly
```

### Test the GenAI summariser

```bash
python -m src.genai.summarizer
```

---

## How to Run the API

> **Note:** The FastAPI backend (`src/api/main.py`) is built in Day 5.
> Run the pipeline and model training first, then start the API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## How to Launch Streamlit

> **Note:** The Streamlit dashboard (`app/streamlit_app.py`) is built in Day 6.
> Ensure the pipeline, model, and API are running first, then launch:

```bash
streamlit run app/streamlit_app.py
```

Dashboard: [http://localhost:8501](http://localhost:8501)

---

## How to Generate Power BI Screenshots Locally

1. Run the pipeline to generate `data/processed/communications_processed.csv`
2. Open Power BI Desktop (free download from Microsoft)
3. Open `dashboard/powerbi/Customer_Communication_Intelligence.pbix`
4. Connect to the CSV file and refresh data
5. Export pages: **File → Export → Export to PDF or PNG**
6. Save screenshots to `assets/screenshots/powerbi/`

See `dashboard/powerbi/README.md` for full instructions.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run individual test modules
python -m pytest tests/test_data.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_anomaly.py -v
python -m pytest tests/test_genai.py -v
```

**Current test count:** 158 tests across 4 modules, all passing.

---

## MVP Limitations

- Synthetic data only — model performance will differ on real data
- No authentication on API endpoints
- DuckDB is local-only — not suitable for multi-user production use
- Single model (Random Forest) — no ensemble or deep learning
- GenAI stub produces templated summaries — not as nuanced as real LLM output
- No real-time streaming — batch processing only
- Power BI integration is screenshots-only, no live connection

---

## Resume-Ready Project Summary

> Built a full-stack Customer Communication Intelligence Platform demonstrating Python, SQL-style analytics (DuckDB), machine learning (Random Forest classifier with 26 engineered features), anomaly detection (z-score, week-over-week trend analysis), GenAI-assisted summarisation (OpenAI API with stub fallback), and a FastAPI + Streamlit production-style architecture. Implemented a modular, config-driven codebase with 100+ unit tests, Codex-reviewed pull requests, and a GitHub-ready repo structure suitable for regulated industry portfolio demonstration.

---

## Future Improvements

- Replace synthetic data with anonymised real datasets
- Add authentication (API keys or OAuth) to FastAPI endpoints
- Deploy to cloud (Azure App Service or AWS Lambda for API, Streamlit Cloud for dashboard)
- Add a CI/CD pipeline (GitHub Actions) with automated test runs on PR
- Extend to multi-model ensemble with explainability (SHAP values)
- Add real-time scoring via a message queue (e.g. Azure Service Bus)
- Supabase/PostgreSQL backend for multi-user production use
- Add email/Slack alerting for anomaly spikes
- Expand GenAI to support conversation-style case review

---

*Built as a portfolio project — GenAIFinance · 2026*
