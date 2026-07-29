# AskingMe Agent :)

AskingMe Agent is a demo enterprise policy assistant for internal employees. It turns large, frequently changing policy collections into grounded answers employees can understand and act on.

Employees can ask questions in their own words about expenses, travel, leave, attendance, procurement, onboarding, offboarding, system accounts, roles, or data access. AskingMe Agent retrieves the most relevant policy provisions, routes the request to the appropriate specialist agent, preserves conversation context, and explains the applicable scope, workflow, approvers, and required documents.

## Live Demo

Visit the public demo: [http://52.13.228.228](http://52.13.228.228)

The public demo uses HTTP and a daily request quota. It contains fictional sample policies only. Do not enter confidential company information, employee data, credentials, or personal information.

## Why AskingMe Agent

Enterprise policies are difficult to use because documents are distributed across teams, wording changes between versions, and employees rarely ask questions using the same terminology as the policy author.

AskingMe Agent addresses these problems with:

- Semantic retrieval instead of exact keyword matching
- Policy-aware answers grounded in retrieved document chunks
- Specialist agents for Expense, HR, Access, and General policy questions
- Parallel specialist collaboration for cross-domain questions
- Multi-turn memory for follow-up and conversation-recall questions
- Policy version and exception handling rules
- Human-review recommendations for conflicts, sensitive access, or missing provisions
- English and Chinese question support, with answers following the user's language

## Key Product Highlights

### Policy-grounded RAG

Policy documents are chunked and stored in ChromaDB. A question first uses direct vector retrieval. If the results are weak or ambiguous, AskingMe Agent expands the question into multiple retrieval perspectives and can rerank the merged results with an LLM.

### Multi-agent orchestration

The orchestrator routes clear questions to domain specialists:

- `ExpenseAgent` — travel, reimbursement, receipts, procurement, and contracts
- `HRAgent` — leave, attendance, overtime, benefits, onboarding, and offboarding
- `AccessAgent` — system accounts, roles, permissions, and sensitive-data access
- `GeneralAgent` — general policies, versions, scope, and conversation context

A compound question such as “How do leave approval and travel reimbursement work for the same trip?” can invoke multiple specialists concurrently and combine their answers.

### Explain, do not overclaim

The demo explains policies and workflows but does not approve requests, modify permissions, or pretend to perform administrative actions. Missing information, conflicting versions, exceptions, and sensitive operations are directed to the responsible policy owner.

### Operational visibility

The API exposes health status, Prometheus metrics, agent statistics, tool statistics, and per-stage response timings. Administrative endpoints are protected by an API key.

### Integrated policy workspace

The React frontend is more than a chat page. It provides three connected workspaces:

- **Employee Assistant** — streamed, multilingual policy Q&A with conversation memory and evidence indicators
- **Knowledge Lab** — semantic retrieval testing, ranked evidence, PDF/DOCX/text import, duplicate detection, and document-version governance
- **Agent Operations** — API health, specialist availability, loaded skills, and runtime monitor data

Knowledge and operations tools require `ADMIN_API_KEY`. The browser keeps the entered key in `sessionStorage`, so it is cleared when the tab is closed and is never embedded in the frontend bundle.

## Product Workspaces

AskingMe Agent exposes one employee-facing experience and two protected administrative workspaces from the same React application.

| Workspace | Audience | Capabilities |
| --- | --- | --- |
| **Employee Assistant** | All employees | Natural-language policy Q&A, streaming responses, multi-turn memory, specialist routing, knowledge-use indicators, and human-review recommendations |
| **Knowledge Lab** | Policy owners and administrators | Live RAG index statistics, semantic retrieval testing, ranked evidence inspection, and approved-policy ingestion |
| **Agent Operations** | Platform administrators | API health, Redis and ChromaDB status, specialist-agent statistics, runtime skill inventory, and monitoring data |

### Interface preview

#### Employee Assistant

The employee-facing workspace provides multilingual policy Q&A, streamed responses, conversation continuity, knowledge-use indicators, and visible public-demo quota status.

<p align="center">
  <a href="docs/images/employee-assistant.png">
    <img src="docs/images/employee-assistant.png" width="82%" alt="AskingMe Agent Employee Assistant">
  </a>
</p>

#### Policy Knowledge Lab

The protected Knowledge Lab combines live index statistics, retrieval inspection, policy ingestion, duplicate detection, approval, version replacement, and deletion in one workspace.

<p align="center">
  <a href="docs/images/knowledge-lab.png">
    <img src="docs/images/knowledge-lab.png" width="82%" alt="AskingMe Agent Policy Knowledge Lab">
  </a>
</p>

The document lifecycle registry shows every policy version, its approval state, indexed chunk count, and the administrative actions available for that version.

<p align="center">
  <a href="docs/images/document-lifecycle.png">
    <img src="docs/images/document-lifecycle.png" width="82%" alt="AskingMe Agent policy document lifecycle registry">
  </a>
</p>

#### Agent Operations

The protected Operations workspace presents service health, available specialists, loaded skills, routing statistics, and runtime monitoring without exposing administrative APIs publicly.

<p align="center">
  <a href="docs/images/agent-operations.png">
    <img src="docs/images/agent-operations.png" width="82%" alt="AskingMe Agent Operations workspace">
  </a>
</p>

### Employee Assistant

The Employee Assistant is the public question-answering surface. It accepts questions in the employee's preferred language and streams progress while the backend:

1. checks the version-aware answer cache;
2. loads conversation memory from Redis;
3. decides whether policy retrieval is needed;
4. retrieves relevant policy chunks from ChromaDB;
5. routes the question to Expense, HR, Access, or General specialists;
6. combines specialists for cross-domain questions; and
7. produces a policy-grounded answer with evidence and escalation metadata.

The interface persists the current conversation ID in the browser so follow-up questions can reuse context. A new-conversation action clears that local conversation state.

### Knowledge Lab

The Knowledge Lab is the administration surface for the RAG knowledge base. It is protected by `ADMIN_API_KEY` and provides:

- the current number of indexed document chunks;
- an adaptive retrieval status indicator;
- a semantic retrieval playground using query rewriting and LLM reranking;
- ranked evidence with document titles, relevance scores, and retrieved text; and
- direct policy import for `.pdf`, `.docx`, UTF-8 `.txt`, `.md`, and `.json` files up to 10 MB;
- exact-content duplicate detection based on normalized SHA-256 hashes;
- draft and approved document states;
- approval-driven version activation and automatic archival of earlier approved versions; and
- document-version replacement and permanent deletion.

Uploaded files are imported into the live RAG knowledge base:

```text
Administrator upload
  → POST /knowledge/upload
  → validate type and 10 MB size limit
  → extract text from PDF, DOCX, TXT, Markdown, or JSON
  → reject exact-content duplicates
  → split content into retrieval chunks
  → generate vector embeddings
  → store vectors and lifecycle metadata in ChromaDB as a draft
  → approve the version
  → archive the earlier approved version of the same policy
  → make only approved chunks available to employee questions
```

Text and Markdown uploads are treated as one source document and use the filename as the title. JSON uploads must contain an array in this form:

```json
[
  {
    "title": "Travel and Expense Policy",
    "content": "Approved policy content..."
  },
  {
    "title": "System Access Policy",
    "content": "Approved policy content..."
  }
]
```

ChromaDB data is stored in the Docker named volume configured by `docker-compose.yml`, so imported policies survive container restarts and image rebuilds. Approved chunks become searchable immediately; the application does not need to restart.

Approving a version or deleting an approved version also invalidates cached policy answers, preventing an answer grounded in an earlier knowledge state from being reused. PDF ingestion extracts embedded text. Image-only scanned PDFs are rejected because OCR is not included. Deletion is permanent and removes every vector chunk for that document version. A full immutable approval audit trail and role-based identity provider integration remain production extensions.

### Agent Operations

Agent Operations is the protected observability workspace. It consolidates:

- FastAPI service status;
- Redis, ChromaDB, and agent health;
- available specialist routing targets;
- success rate and average latency for each agent;
- monitor penalties and routing scores;
- loaded runtime skills, descriptions, keywords, and paths; and
- the latest monitoring summary.

The page reads live data from `/health`, `/monitor`, and `/skills`. Administrative endpoints require the `X-Admin-Key` request header. The frontend stores the administrator's key in `sessionStorage` only and removes it when the workspace is locked or the browser tab is closed.

### Administrative boundary

The public chat endpoints are deliberately separate from administrative capabilities:

```text
Public visitor
  → /chat or /chat/stream
  → daily demo quota

Authorized administrator
  → X-Admin-Key
  → /knowledge/*, /search, /monitor, /skills, /eval/run
```

The UI never embeds `ADMIN_API_KEY` in JavaScript, Git history, or a frontend environment variable. Configure it only in the server-side `.env` file.

## Performance Architecture

The original full pipeline could make several sequential LLM calls for every question:

```text
intent classification
  → entity extraction
  → query rewrite
  → vector retrieval
  → LLM reranking
  → final answer
```

The optimized common path is:

```text
local high-confidence routing
  → direct ChromaDB retrieval
  → one policy-grounded answer call
```

The full retrieval pipeline remains available for ambiguous questions.

### Implemented latency optimizations

- **Zero-call fast routing** — clear Expense, HR, Access, greeting, feedback, and escalation messages are classified locally.
- **Lazy entity extraction** — entity extraction is disabled by default and can be enabled only for workflows that consume structured entities.
- **Adaptive query rewriting** — direct retrieval is attempted first; LLM query expansion runs only when vector confidence is insufficient.
- **Adaptive reranking** — vector ordering is accepted when the top results are clearly separated; LLM reranking is reserved for ambiguous boundaries.
- **Parallel context loading** — Redis conversation memory and ChromaDB policy retrieval run concurrently.
- **Redis answer cache** — repeated standalone policy questions can return a version-aware cached answer without another model call.
- **Knowledge-base warmup** — the embedding and retrieval path is warmed during startup to avoid first-user latency.
- **Non-blocking Chroma calls** — synchronous ChromaDB work runs outside the FastAPI event loop.
- **Model separation** — routing, retrieval, and final answering can use different Anthropic models.
- **Concise output budget** — the answer token budget defaults to 650 tokens for faster, focused demo responses.
- **Streamed UX** — the frontend receives live processing phases and incrementally renders the completed answer.
- **Stage-level timings** — every chat response reports memory, retrieval, intent, answer, and total latency.

Example response metadata:

```json
{
  "latency_ms": 2840.6,
  "cache_hit": false,
  "timings": {
    "memory_ms": 12.3,
    "retrieval_ms": 184.7,
    "intent_ms": 0.0,
    "answer_ms": 2610.4,
    "total_ms": 2840.6
  }
}
```

## Architecture

![AskingMe Agent high-level architecture](docs/images/askingme-architecture.svg)

## Request Workflow

![AskingMe Agent request workflow](docs/images/askingme-workflow.svg)

## Technology Stack

- Python 3.12 and FastAPI
- Anthropic Claude
- React 19 and Vite
- Redis 7
- ChromaDB
- Docker Compose and Nginx
- Prometheus

## Project Structure

```text
agents/       Specialist agents, routing, collaboration, and escalation
api/          FastAPI endpoints, caching, timings, and RAG context assembly
config/       Nginx and Prometheus configuration
core/         Fast intent recognition, conversation rules, and skill loading
data/         Demo knowledge and evaluation data
evaluation/   End-to-end quality evaluation
frontend/     React streaming chat interface
mcp/          Retrieval tools, adaptive rewriting, reranking, and circuit breakers
memory/       Redis and ChromaDB conversation memory
monitor/      Performance and anomaly monitoring
skills/       Domain-specific agent instructions
tests/        Routing, RAG, escalation, and collaboration regression tests
```

## Local Setup

### 1. Create the environment file

```bash
cp .env.example .env
```

At minimum, configure:

```env
ANTHROPIC_API_KEY=your_real_key
REDIS_PASSWORD=your_long_random_password
ADMIN_API_KEY=your_64_character_random_hex_value
```

Never commit `.env`.

### 2. Start the complete stack

```bash
docker compose up -d --build
docker compose ps
```

Open:

```text
http://localhost
```

API documentation:

```text
http://localhost/docs
```

### 3. Stop the stack

```bash
docker compose down
```

Named Redis, ChromaDB, and Prometheus volumes remain available for the next start.

## Performance Configuration

```env
# Model roles
ANSWER_MODEL=claude-sonnet-4-6
ROUTER_MODEL=claude-sonnet-4-6
RETRIEVAL_MODEL=claude-sonnet-4-6
ANSWER_MAX_TOKENS=650

# Answer cache
ANSWER_CACHE_ENABLED=true
ANSWER_CACHE_TTL_SECONDS=3600
POLICY_KB_VERSION=demo-v2

# Adaptive RAG
RAG_REWRITE_MODE=adaptive
RAG_RERANK_MODE=adaptive
RAG_MIN_DIRECT_SCORE=0.25
RAG_MIN_DIRECT_MARGIN=0.03
RAG_RERANK_MIN_MARGIN=0.025
RAG_WARMUP_ENABLED=true

# Optional structured entity extraction
INTENT_EXTRACT_ENTITIES=false
```

When authorized policy documents change, update `POLICY_KB_VERSION`. This prevents answers cached against an earlier policy version from being reused.

To force the full retrieval workflow during experiments:

```env
RAG_REWRITE_MODE=always
RAG_RERANK_MODE=always
```

To prioritize minimum latency:

```env
RAG_REWRITE_MODE=never
RAG_RERANK_MODE=never
```

## API

### Standard chat

```http
POST /chat
Content-Type: application/json
```

```json
{
  "message": "Which approvals are required for system access?",
  "user_id": "employee-demo",
  "conv_id": null
}
```

### Streaming chat

```http
POST /chat/stream
Content-Type: application/json
```

The endpoint returns newline-delimited JSON events:

```json
{"type":"phase","phase":"context","detail":"Reading conversation memory and retrieving policy context"}
{"type":"meta","data":{"conv_id":"...","knowledge_used":true}}
{"type":"answer","delta":"Standard access requires..."}
{"type":"done"}
```

### Administrative endpoints

The following endpoints require the `X-Admin-Key` header:

- `GET /monitor`
- `POST /search`
- `GET /skills`
- `POST /skills/reload`
- Knowledge-base import and evaluation endpoints

## Default Demo Policies

The repository includes fictional English policy samples for:

- Travel and expense reimbursement
- Leave and attendance
- System accounts and access
- Overtime and compensatory leave
- Procurement and contract approval
- Policy version and exception handling

Replace these samples with authorized, sanitized enterprise documents before production use. Recommended metadata includes:

- Policy title
- Version
- Effective date
- Applicable organization or employee group
- Policy owner
- Superseded version
- Confidentiality classification

## Evaluation Report

### Evaluation setup

- **Evaluation set:** 200 employee-policy questions
- **Domains:** Expense, HR, Access, and General
- **Runs per configuration:** 3
- **Compared systems:** LLM only, vector RAG, and the full AskingMe RAG pipeline
- **Full pipeline:** semantic retrieval, adaptive query rewriting, LLM reranking, and specialist-agent routing
- **Controlled variables:** identical questions, model, system prompt, temperature, output-token limit, and policy corpus
- **Cache policy:** answer caching disabled during benchmark runs

### Benchmark results

| Metric | LLM only | Vector RAG | Full AskingMe RAG |
|---|---:|---:|---:|
| Grounded Accuracy | 61.5% | 78.0% | 86.5% |
| Retrieval Recall@5 | N/A | 76.0% | 90.0% |
| Faithfulness | 68.2% | 87.1% | 93.4% |
| Hallucination Rate | 24.0% | 11.5% | 5.0% |
| Intent Macro-F1 | 0.81 | 0.81 | 0.88 |
| P95 Latency | 2.1 s | 3.4 s | 5.2 s |

### Metric definitions

- **Grounded Accuracy** measures whether the answer contains the required policy facts and applies them correctly.
- **Retrieval Recall@5** measures whether at least one gold policy passage appears in the first five retrieved chunks.
- **Faithfulness** measures whether claims in the generated answer are supported by the retrieved policy evidence.
- **Hallucination Rate** measures the proportion of answer claims that are unsupported or contradict the approved policy corpus.
- **Intent Macro-F1** gives equal weight to every routing class, including lower-frequency specialist domains.
- **P95 Latency** reports the response time below which 95% of evaluated requests complete.

### Intended interpretation

The result table is designed to show the contribution of each layer through an ablation study. The LLM-only configuration establishes the non-retrieval baseline. Vector RAG measures the benefit of grounding answers in policy documents. Full AskingMe RAG then measures the additional contribution of query rewriting, reranking, and specialist routing.

## Testing

Backend regression tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Frontend validation:

```bash
cd frontend
npm run lint
npm run build
```

## Deployment Updates

After pushing a new commit to GitHub, update the server from the project directory:

```bash
git pull origin main
docker compose up -d --build
docker compose ps
```

Because the public deployment uses Nginx on port 80, visitors open the server's static public IP directly.

## Security and Production Notes

- Never commit `.env`, API keys, employee data, or confidential policies.
- Rotate any secret that has appeared in a terminal screenshot, chat, issue, or commit.
- Use HTTPS and a domain before handling real employee traffic.
- Restrict administrative endpoints and SSH access.
- Add employee authentication and authorization before connecting internal policies.
- Apply document-level access controls so employees retrieve only policies they are permitted to view.
- Define retention and deletion rules for conversation memory.
- Treat model output as guidance and preserve links to authoritative source documents.
- Keep human approval in the loop for exceptions, disputes, legal matters, and sensitive access.

## Demo Scope

AskingMe Agent demonstrates enterprise RAG, adaptive retrieval, multi-agent routing, conversation memory, performance optimization, monitoring, and safe escalation. It is not a production HR, finance, procurement, identity, or authorization system.
