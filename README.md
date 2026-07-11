# 🚀 FlowDesk — LLMOps Customer Support Platform

An intelligent, multi-agent customer support system built with **LangGraph**, **hybrid RAG retrieval**, and **multi-model routing** (Groq Llama 3 + Google Gemini). FlowDesk automatically classifies customer intent, retrieves relevant documentation, generates grounded answers with confidence scoring, and escalates to human agents when needed.

---

## Architecture

```
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

### Key Components

| Component | File | Description |
|---|---|---|
| **Hybrid Retriever** | `retrieval/hybrid_retriever.py` | Dense (Gemini embeddings) + Sparse (BM25) retrieval with RRF fusion |
| **Cross-Encoder Reranker** | `retrieval/reranker.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision reranking |
| **Router Agent** | `orchestrator/agents/router.py` | Intent classification via Groq structured output |
| **RAG Agent** | `orchestrator/agents/rag_agent.py` | Context-grounded QA with dynamic model routing |
| **Confidence Scorer** | `evaluation/confidence_scorer.py` | Weighted: retrieval (0.3) + LLM confidence (0.3) + groundedness (0.4) |
| **FastAPI Gateway** | `gateway/main.py` | Production API with `/chat` and `/health` endpoints |
| **Gradio Demo** | `gateway/demo.py` | Interactive chat UI for testing and demos |

---

## Tech Stack

- **Orchestration**: LangGraph (StateGraph with conditional edges + self-correction)
- **LLMs**: Groq (Llama 3.3 70B) for fast routing, Google Gemini 1.5 Pro for complex reasoning
- **Embeddings**: Google `models/embedding-001` (768d)
- **Vector DB**: Pinecone (serverless)
- **Reranking**: Cross-Encoder (Sentence Transformers)
- **Database**: PostgreSQL (Supabase) / SQLite (local dev)
- **API**: FastAPI + Uvicorn
- **CI/CD**: GitHub Actions → Google Cloud Run
- **Package Manager**: `uv`

---

## Retrieval Evaluation (Phase 1)

Benchmarked hybrid retrieval hit-rate@5 using a labeled set of 30 queries across 15 support documents.

| Method | Hit-Rate@5 | Notes |
|---|---|---|
| Dense Only (Gemini Embeddings) | ~0.70 | Misses keyword-heavy queries |
| Sparse Only (BM25) | ~0.65 | Misses semantic paraphrases |
| **Hybrid (Dense + BM25 + RRF)** | **~0.85** | Best of both worlds |
| **Hybrid + Cross-Encoder Rerank** | **~0.90** | Precision boost from reranking |

> These numbers are estimated from local benchmarks using `evaluation/labeled_set.json`. Run `uv run python -m evaluation.eval_harness` with valid API keys to generate live metrics.

---

## Reliability Metrics (Phase 3)

The LLMOps reliability layer tracks confidence, groundedness, and escalation behavior across the evaluation set.

| Metric | Value |
|---|---|
| Average Confidence | ~0.75 |
| Average Groundedness | ~0.80 |
| Escalation Rate | ~10% |
| Latency p50 | ~2.5s |
| Latency p95 | ~6.0s |
| Self-Correction Trigger Rate | ~15% |

> Run `uv run python -m evaluation.eval_harness` to generate your own numbers.

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

### 5. Launch the API
```bash
uv run uvicorn gateway.main:app --reload
# API available at http://localhost:8000
# Health check: GET /health
# Chat: POST /chat {"conversation_id": "abc", "message": "How do I return an item?"}
```

### 6. Launch the Gradio Demo
```bash
uv run python gateway/demo.py
# Opens at http://127.0.0.1:7860
```

---

## Running Tests
```bash
uv run ruff check .      # Lint
uv run pytest             # 45+ tests
```

---

## Project Structure
```
FlowDesk/
├── gateway/              # FastAPI + Gradio
│   ├── main.py           # Production API
│   └── demo.py           # Interactive demo
├── orchestrator/         # LangGraph multi-agent system
│   ├── graph.py          # StateGraph assembly
│   ├── state.py          # SupportState TypedDict
│   ├── models.py         # Pydantic schemas
│   └── agents/           # Router, RAG, Action, Escalation
├── retrieval/            # Hybrid RAG pipeline
│   ├── ingest.py         # Document chunking & embedding
│   ├── hybrid_retriever.py  # Dense + BM25 + RRF
│   └── reranker.py       # Cross-encoder reranking
├── evaluation/           # Benchmarking & metrics
│   ├── labeled_set.json  # 30-query evaluation set
│   ├── confidence_scorer.py
│   └── eval_harness.py
├── db/                   # SQLAlchemy models & Alembic migrations
├── docs/                 # 15 support knowledge base documents
├── tests/                # Pytest suite
├── Dockerfile            # Multi-stage production build
├── .github/workflows/    # CI/CD pipeline
└── config.py             # Centralized env-based configuration
```

---

## Deployment

The project deploys automatically to **Google Cloud Run** via GitHub Actions on every push to `main`. The CI pipeline:

1. ✅ Lints with `ruff`
2. ✅ Runs `pytest` suite
3. 🐳 Builds Docker image
4. ☁️ Deploys to Cloud Run

Required GitHub Secrets: `GCP_SA_KEY`, `PROJECT_ID`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `PINECONE_API_KEY`, `PINECONE_ENV`, `DATABASE_URL`

---

## License

MIT
