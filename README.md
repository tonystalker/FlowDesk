# 🚀 FlowDesk

An intelligent, multi-agent customer support platform built with **LangGraph**, **hybrid RAG retrieval**, and **multi-model routing** (Groq Llama 3 + Google Gemini). FlowDesk automatically classifies customer intent, retrieves relevant documentation, generates grounded answers with confidence scoring, and seamlessly escalates to human agents when necessary.

---

## Architecture

```text
User Query
    │
    ▼
┌──────────┐
│  Router  │  ← Groq Llama 3.3 (intent classification)
│  Agent   │
└────┬─────┘
     │
     ├── faq ──────────► ┌───────────┐
     │                   │ RAG Agent │ ← Hybrid Search + Cross-Encoder Reranking
     │                   │           │   → Groq (short) / Gemini (complex/retry)
     │                   └─────┬─────┘
     │                         │
     │                    confidence < 0.6?
     │                    ┌────┴────┐
     │                    │  Yes    │  No
     │                    │ Retry   │  → ✅ Return Answer
     │                    │ (wider  │
     │                    │  search,│
     │                    │  switch │
     │                    │  model) │
     │                    └─────────┘
     │
     ├── action ───────► ┌──────────────┐
     │                   │ Action Agent │ ← Tool-calling (orders, refunds, account)
     │                   └──────────────┘
     │
     └── complex ──────► ┌────────────────────┐
                         │ Escalation Agent   │ ← Human handoff
                         └────────────────────┘
```

## Key Features

- **Hybrid Retrieval Pipeline**: Combines dense semantic search (Gemini embeddings) with sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF), followed by Cross-Encoder reranking for high precision.
- **Dynamic Model Routing**: Routes latency-sensitive or simple queries to Groq (Llama 3.3), while reserving Gemini for complex reasoning and conversational retries.
- **LLMOps & Telemetry**: Every interaction is logged to a PostgreSQL database, tracking confidence scores, groundedness metrics, and explicit user feedback (thumbs up/down) for continuous improvement.
- **Graceful Degradation**: Built-in fallbacks ensure the system remains operational (e.g., seamlessly dropping to sparse-only search if the embedding API is unreachable).

## Tech Stack

- **Orchestration**: LangGraph (StateGraph with conditional edges + self-correction)
- **LLMs**: Groq (Llama 3.3 70B) & Google Gemini (3.5 Flash / 2.5 Pro)
- **Embeddings**: Google `models/gemini-embedding-2`
- **Vector Database**: Pinecone (Serverless)
- **Reranking**: Sentence Transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Database**: PostgreSQL (Supabase) / SQLite (Local Dev)
- **API**: FastAPI + Uvicorn + Server-Sent Events (SSE) Streaming
- **UI**: Gradio
- **CI/CD**: GitHub Actions → Google Cloud Run

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone & Install
```bash
git clone https://github.com/tonystalker/FlowDesk.git
cd FlowDesk
uv sync
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys:
#   GEMINI_API_KEY, GROQ_API_KEY, PINECONE_API_KEY, PINECONE_ENV, DATABASE_URL
```

### 3. Run Database Migrations
```bash
uv run alembic upgrade head
```

### 4. Ingest Support Documents
```bash
uv run python -m retrieval.ingest
```

### 5. Launch the API & UI
**Start the REST API:**
```bash
uv run uvicorn gateway.main:app --reload
# Available at http://localhost:8000
```

**Start the Interactive Chat UI:**
```bash
uv run python -m gateway.demo
# Opens at http://127.0.0.1:7860
```

---

## Testing & Evaluation

The platform includes a comprehensive test suite and evaluation harness to benchmark retrieval accuracy and agent reliability.

```bash
# Run the test suite (50+ tests)
uv run pytest -v

# Run the linter
uv run ruff check .

# Run the evaluation harness for retrieval metrics
uv run python -m evaluation.eval_harness
```

---

## Project Structure
```text
FlowDesk/
├── gateway/              # FastAPI + Gradio UI
├── orchestrator/         # LangGraph multi-agent system & state
│   └── agents/           # Router, RAG, Action, and Escalation agents
├── retrieval/            # Hybrid RAG pipeline (Ingest, Dense/Sparse, Reranking)
├── evaluation/           # Benchmarking & confidence scoring
├── db/                   # SQLAlchemy models & Alembic migrations
├── docs/                 # Support knowledge base documents
├── tests/                # Pytest suite
├── Dockerfile            # Multi-stage production build
├── .github/workflows/    # CI/CD pipeline
└── config.py             # Centralized environment configuration
```

---

## Deployment

The project is configured to deploy automatically to **Google Cloud Run** via GitHub Actions on every push to the `main` branch. 

The CI pipeline automatically:
1. Lints the codebase with `ruff`
2. Runs the full `pytest` suite
3. Builds the Docker container
4. Deploys to Cloud Run

*Required GitHub Secrets:* `GCP_SA_KEY`, `PROJECT_ID`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENV`, `DATABASE_URL`
