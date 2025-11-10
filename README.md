# Testing Agent (No LangFlow) — MVP

This repo provides a local-first testing agent that evaluates external AI systems via **black-box API**.  
The user supplies **header.json** and **body.json**, and the system uses an **LLM (Gemini)** plus heuristics to infer:
- where to inject the **prompt** in the request (JSONPath),
- where to extract the **returned message** from the response (JSONPath),
- error-detection rules (status ranges, error fields/regex).

The user selects **automatable/API metrics** (by **Standard** or **Manual**), executes a dataset of prompts, collects artifacts, computes metric scores, and exports **CSV/PDF**.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env for GEMINI_API_KEY, optional GEMINI_MODEL, DATABASE_URL

uvicorn backend.app:app --reload
```

API will run on http://127.0.0.1:8000 (Swagger at `/docs`).

## Folder Layout

```
backend/
  app.py                 # REST API
  config.py              # env, settings
  db.py                  # engine/session
  models.py              # SQLAlchemy ORM
  schemas.py             # Pydantic models
  mapping_service.py     # LLM+heuristics → JSONPaths & error rules
  executor.py            # sends requests; stores artifacts
  evaluator.py           # metrics: function + LLM-judge
  reporting.py           # CSV/PDF exports
  clients/gemini_client.py
  utils/jsonpath_utils.py
data/
  standards.json
  metrics.json
  datasets/sample.jsonl
storage/
  artifacts/             # per-run requests/responses
  reports/               # PDFs/CSVs
migrations/              # (optional Alembic migrations)
tests/                   # (stubs)
```

## Core Endpoints

- `POST /sut` — create/update a system (headers/body/endpoint)
- `GET /sut/{system_id}`
- `POST /mapping/detect` — propose prompt/response paths + error rules
- `POST /plan` — create run (standard/manual + metrics list + dataset ref)
- `POST /run` — execute dataset → artifacts
- `POST /evaluate` — compute metrics → results
- `GET /runs` — list runs
- `GET /run/{run_id}/results`
- `GET /reports/{run_id}/{type}` — download `csv|pdf`

New endpoints (data model upgrade):
- `POST /datasets` — upsert a dataset (profile)
- `POST /datasets/items` — upsert a dataset item
- `POST /standards/metrics` — link a standard to a metric (with `required` flag)
- `POST /metrics/update` — update a metric’s `unit`, `aggregation`, `applicability`

## Standards & Metrics
See `data/standards.json` and `data/metrics.json`. Edit later or migrate to Postgres using Alembic.

### Notes
- LLM features (mapping refine and LLM-judge metrics) gracefully fallback if `GEMINI_API_KEY` is not set; the app remains testable offline.
- To use a registered dataset in `/plan`, first insert a dataset and items (see below) and set `dataset: {"type":"dataset_id","value":"<id>"}`.

## Swap SQLite → Postgres
Set `DATABASE_URL` in `.env`, then run Alembic migrations as needed.

## Load sample dataset (optional)

You can load the sample profile and items into the DB directly:

```bash
python scripts/load_dataset_from_files.py data/datasets/qa_small_en.profile.json data/datasets/qa_small_en.jsonl
```

Then plan with:

```json
{
  "run_id": "r1",
  "system_id": "sys1",
  "mode": "manual",
  "metric_codes": ["latency_check","reliability_check"],
  "dataset": {"type":"dataset_id","value":"qa_small_en"}
}
```
