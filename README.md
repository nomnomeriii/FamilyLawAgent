# NY Family Law Agent (EECS 6895 Midterm)

This project builds a New York family-law assistant with two RAG paths:

- Procedure Engine: retrieves NY family court forms/instructions and generates filing workflows with source tags.
- Research Engine: retrieves NY case law from CourtListener and generates citation-grounded case summaries.

The current local app is a Streamlit UI (`app_streamlit.py`) backed by modular Python code in `family_law_agent/`.

## Team Member
- Chih-Hsin Chen (cc5240)
- Leah Li (ql2481)
- Yanhao Bai (yb2630)

## Repository Layout

- `app_streamlit.py`: main Streamlit app.
- `family_law_agent/procedure.py`: local forms ingestion + procedure answer generation.
- `family_law_agent/research.py`: CourtListener retrieval, reranking, quote extraction, synthesis prompt.
- `family_law_agent/safety.py`: safety classifier for harmful/illegal requests.
- `scripts/build_procedure_db.py`: CLI to build the Chroma DB from local forms.
- `requirements_streamlit.txt`: dependencies for local app run.
- `NY Family Law Forms/`: source documents for Procedure Engine.

## Prerequisites

- OpenAI API key.
- CourtListener token (recommended for research retrieval quality).

## Setup (Recommended: Virtual Environment)

Run from project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_streamlit.txt
```

## Configure API Keys (.env)

Create and edit a local `.env` file:

```bash
cp .env.example .env
```

Then set:

```dotenv
OPENAI_API_KEY=sk-...
COURTLISTENER_TOKEN=...
```

## Build Procedure Database

```bash
python scripts/build_procedure_db.py \
  --forms-dir "./NY Family Law Forms" \
  --db-dir "./db_procedure"
```

## Run the App

```bash
streamlit run app_streamlit.py
```

In the Streamlit sidebar, provide keys only if you did not set them in `.env`:

- OpenAI API Key (required unless set as `OPENAI_API_KEY`).
- CourtListener Token (recommended unless set as `COURTLISTENER_TOKEN`).

## Quick CourtListener Check (Optional)

Use this if the Research Engine returns no results:

```bash
python scripts/check_courtlistener.py --token "<YOUR_TOKEN>"
```

If `COURTLISTENER_TOKEN` is set in `.env`, you can also run:

```bash
python scripts/check_courtlistener.py
```

If status is `401/403`, the token is invalid/expired or malformed.
If status is `200` with non-zero `result_count`, API connectivity is working.

## App Behavior

- Enforces NY jurisdiction gate.
- Runs safety classifier before engine execution.
- Uses chat-style intake (case type selection -> question).
- Executes Procedure and Research engines per query.
- Shows a 3-tab workspace: `Workflow Checklist` / `Case Research` / `Draft Outline`.
- Supports `Export Filing Packet (.md)` for checklist + research + draft outline.
- Falls back to general informational guidance if citation quality is weak.

## Evaluation Datasets and Scripts

Dataset scaffolds:

- `data/eval/procedure_workflow_test.jsonl`
- `data/eval/case_retrieval_test.jsonl`
- `data/eval/ny_bar_benchmark.jsonl` (optional placeholder)

Metric scripts:

- `scripts/eval_procedure.py`
- `scripts/eval_research.py`
- `scripts/generate_eval_predictions.py`

### Generate Predictions

```bash
python scripts/generate_eval_predictions.py --max-cases 0
```

Resume mode (generate only missing IDs from existing prediction files):

```bash
python scripts/generate_eval_predictions.py --resume --max-cases 0
```

### Run Evaluation

```bash
python scripts/eval_procedure.py \
  --gold data/eval/procedure_workflow_test.jsonl \
  --pred data/eval/procedure_predictions.jsonl \
  --out data/eval/procedure_report.json
```

```bash
python scripts/eval_research.py \
  --gold data/eval/case_retrieval_test.jsonl \
  --pred data/eval/research_predictions.jsonl \
  --k 5 \
  --out data/eval/research_report.json
```

### Re-run Single Case (Example: `CR-001` only)

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("data/eval/research_predictions.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("id") != "CR-001"]
p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print("Removed CR-001 from predictions. Remaining rows:", len(rows))
PY

python scripts/generate_eval_predictions.py --resume --max-cases 1 --skip-procedure
```

For a single Procedure case (example `WF-001`), remove `WF-001` from `data/eval/procedure_predictions.jsonl`, then run:

```bash
python scripts/generate_eval_predictions.py --resume --max-cases 1 --skip-research
```


