# Final Price Agent Implementation Map

The primary application is an English-language Final Price AI Agent combining rule retrieval with a deterministic Decimal calculation tool. The original enterprise policy agent remains available as a compatibility mode.

| Requirement | Implementation | Verifiable behavior |
|---|---|---|
| Final-price arithmetic | `rag/calculator.py`, `rag/llm.py` | Bounded expression evaluation, half-up rounding, and computed intermediate steps |
| Stateful LangGraph workflow and conditional rewriting | `rag/graph.py` | Analysis, hybrid retrieval, reranking, generation, and bounded retries for weak evidence |
| LangChain integration | `rag/chunking.py`, `rag/llm.py` | Document metadata and ChatPromptTemplate |
| Elasticsearch BM25, dense retrieval, and RRF | `rag/elasticsearch_store.py` | Concurrent retrieval branches with shared prefilters and rank fusion |
| Structure-aware chunking and formula integrity | `rag/chunking.py` | Heading hierarchy, balanced brackets, intact code blocks, and explicit handling of oversized rules |
| Parent references and summary metadata | `rag/chunking.py` | Parent IDs, heading paths, formula types, and deterministic title/heading summaries |
| Cross-Encoder reranking and dynamic Top-K | `rag/models.py`, `rag/graph.py` | Relevance threshold, score gap, and context budget |
| Labeled query/document pairs | `examples/evaluation/pricing.jsonl` | Eight manually labeled English demonstration cases |
| Recall@5, assertion evaluation, and ablations | `evaluation/rag_benchmark.py` | Four configurations; faithfulness and reference-claim coverage reported separately |
| Chat and knowledge administration | `api/main.py`, `frontend/src/App.jsx` | Streaming answers, source citations, filters, imports, and approvals |

## English defaults

The interface, prompts, fallback messages, example rules, evaluation fixtures, and project documentation are in English. The default embedding model is `BAAI/bge-small-en-v1.5`, with its English query instruction. Elasticsearch uses the English text analyzer. The existing reranker supports English. Answers are instructed to use English.

The default index is `askingme-pricing-en-v1`. If an existing `.env` explicitly selects the earlier model or index, update both `EMBEDDING_MODEL` and `ELASTICSEARCH_INDEX` to the values in `.env.example`. Reimport custom rules into the new index. Embeddings from different models must not be mixed, even if their dimensions match. Existing indexes are not automatically deleted or migrated.

English demo conversations use separate browser storage keys. Previously saved conversations remain in their original storage entries. Unicode bracket handling and multilingual input-recognition rules remain available for imported documents and legacy compatibility.

## Implementation choices

RRF runs in application code to avoid dependence on Elasticsearch's server-side RRF licensing and version requirements. The default chunk target is 400 BGE tokens, leaving room within model limits for the question. Oversized formulas produce an actionable error instead of being silently truncated. The reranker checks the actual question/document pair length again.

The example currency remains CNY so the underlying pricing rules and expected calculations are unchanged. A 10% member discount means multiplying the remaining amount by 0.9. Final amounts use decimal round-half-up; the formula examples are explanatory pseudocode.

## Evidence and limitations

The source document's accuracy improvements, recall percentages, 50 ms latency, million-document scale, and 500-query labeling claims have no supporting experiment records in this repository. They are not presented as achieved results. The benchmark produces reports from actual runs. Kafka/Flink integrations are not included; document updates use the authenticated ingestion API.

Offline tests exercise actual LangGraph and FastAPI code with fake external services and models. The real Elasticsearch 8.17.2 contract test has now passed, covering ingestion, hybrid queries, filters, approval, archival, and deletion. See [the verification record](elasticsearch-verification.md). The test uses deterministic vectors, so model inference and end-to-end answer quality still require separate validation. The reproducible runner creates and removes an isolated container, and the test removes its temporary index. Run the benchmark on your own corpus and labels, and retain its configuration and data hashes.

## Calculation accuracy boundary

The final-price tool executes supported model-proposed expressions with Decimal and formats the resulting amount. This removes reliance on generated arithmetic for calculation plans. It does not independently prove that every operand, currency, or eligibility decision matches the retrieved policy. Model-selected explanatory prose is not a tool-verified calculation. The eight new calculator and agent-contract tests cover discount order examples, cents rounding, invalid expressions, evidence references, clarification, and validated answer delivery. Live model-backed evaluation remains necessary.
