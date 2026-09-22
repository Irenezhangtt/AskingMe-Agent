# Elasticsearch Integration Verification

The real Elasticsearch contract test passed on September 22, 2026, against Elasticsearch 8.17.2 in an isolated Docker container. The repeatable test runner also passed and removed its container afterward.

## Environment

- Official image: `docker.elastic.co/elasticsearch/elasticsearch:8.17.2`
- Image digest: `sha256:9fb5d27b49cb8d895209736d8e5d1bea4e0c2232c35874bf4c36628e9c7f7e54`
- Platform: Linux ARM64 through Docker Desktop
- Elasticsearch Python client: 8.17.2
- Single node, 512 MB JVM heap, loopback-only HTTP port
- Unique temporary test index, removed after the test

## Verified behavior

The test writes documents through the production `ElasticsearchKnowledgeBase` implementation and verifies:

1. Index creation and document ingestion with English text fields, dense vectors, dates, and semantic metadata.
2. Real BM25 and kNN search requests combined by application-side RRF.
3. Identical category/date constraints on retrieval; mismatched categories and expired rules return no evidence.
4. Draft documents excluded from search.
5. Approval of a replacement version and archival of the previous version.
6. Retrieval of only the newly approved version after replacement.
7. Document deletion and removal from subsequent search results.
8. Document registry results after deletion.

No production code defect was observed in these tested operations.

## Reproduce

From the repository root, with Docker running and Python dependencies installed:

```bash
.venv/bin/python scripts/test_es_integration.py
```

The runner uses the official image, downloads it if needed, allocates an available loopback port, waits for Elasticsearch health, runs the contract test, and removes its own container in a cleanup block. It does not use existing application indexes or containers. The downloaded image remains available for future runs.

To use an existing dedicated test instance instead:

```bash
ES_TEST_URL=http://127.0.0.1:19200 \
  .venv/bin/python -m unittest discover -s tests -p test_elasticsearch_integration.py -v
```

## Scope

This is a real Elasticsearch integration test, not a mock of the search server. It deliberately uses deterministic three-dimensional embeddings, so it needs no model downloads or LLM credentials. It validates API compatibility, storage, filtering, and document lifecycle behavior. It does not measure BGE semantic retrieval quality, Cross-Encoder relevance, generated-answer accuracy, or production-scale latency. Those remain the responsibility of the model-backed benchmark.
