# Measured golden-set evaluation

Evaluation date: **2026-09-22**. All metrics below are measured on the versioned **synthetic** dataset, not copied from the resume brief. Human annotation and production validation remain pending.

**[500 cases and corpus](../examples/evaluation/golden/)** · **[Machine-readable summary](../evaluation/results/golden-summary.json)** · **[Resume implementation audit](resume-implementation-audit.md)**

## Dataset and protocol

- **500 unique questions**, **25 families × 20 cases**, **156 fictional English policy documents**.
- **390 numerical targets**, computed with an independent rational/integer-cent oracle; **110 clarification targets**.
- Family-separated split: **100 development / 400 evaluation**. The synthetic cases have been inspected during development; the second split is not independent human-held-out evidence.
- Dataset SHA-256: `8f86871c62207e004060a67a7f1c9d0f54eb3e7d114ffddf23607a9a60ea8382`.
- Agent model alias: `claude-sonnet-5`. Provider-default sampling; no deterministic-generation claim.
- The agent benchmark uses reviewed upload text, explicit order facts and the real pricing planner/Decimal tool. It does not measure OCR or retrieval. Fifty cases terminate in local validity/integrity guards; the rest call the model.

## Complete first agent run

| Measure | Observed result |
|---|---:|
| Requests evaluated | 500 / 500 |
| Expected outcome and price/exclusions satisfied | **492 / 500 (98.40%)** |
| Exact currency and amount on numeric cases | **382 / 390 (97.95%)** |
| Expected clarification outcome | **110 / 110** |
| Fully cited expected outcomes | **492 / 500** |
| Provider errors | **0** |
| p50 / p95 request latency | **2650 / 4759 ms** |

The eight failures returned a validation fallback rather than a verified amount. This run did not emit a structured price on any expected-clarification case. Clarification-action scoring does not verify the full semantic quality of the question asked.

| Failed case | Family | Recorded outcome |
|---|---|---|
| `gold-09-10` | stacking_order | Unvalidated model plan; no computed price |
| `gold-16-03` | half_up_rounding | Unvalidated model plan; no computed price |
| `gold-16-06` | half_up_rounding | Unvalidated model plan; no computed price |
| `gold-16-08` | half_up_rounding | Unvalidated model plan; no computed price |
| `gold-16-09` | half_up_rounding | Unvalidated model plan; no computed price |
| `gold-16-10` | half_up_rounding | Unvalidated model plan; no computed price |
| `gold-16-11` | half_up_rounding | Unvalidated model plan; no computed price |
| `gold-18-08` | illustrative_parentheses | Unvalidated model plan; no computed price |

The original raw failed plans were not retained, so schema, syntax and truncation causes cannot be separated retrospectively. Three exploratory retries succeeded; they **do not replace** the original first-pass results. [Failure records](../evaluation/results/initial-agent-failures.json).

## Current condition-explanation version: incomplete model validation

The first answer format often gave arithmetic without explaining exception scope. The current version requires cited `rule_checks` covering eligibility, exclusions, grouping and rounding. It also emits a safe validation-error category for future debugging.

A second 500-case attempt recorded **162 completed responses** and **338 provider errors**. A follow-up minimal request confirmed **insufficient Anthropic API credit**. This incomplete attempt is retained; its error-inclusive pass rate is **not** an estimate of model capability or evidence that the remaining cases passed.

The two model-response validation failures before that service interruption reported `CalculationError` in repeated-threshold cases. The remaining unverified requests must be rerun after service is restored. The current version therefore must not be advertised as having the first run's complete 500-case score.

```bash
# After restoring API credit, retry only service-error/not-run cases under the same code/model/data.
.venv/bin/python -m evaluation.golden.benchmark --mode agent \
  --output reports/golden-500-agent-with-conditions.json \
  --resume --retry-errors --concurrency 6
```

The retry runner preserves prior reports/checkpoint attempts and stops further paid requests on a recognized credit-exhaustion response, while still evaluating local guards.

## Real retrieval ablations

Actual **Elasticsearch 8.17.2**, **BGE small English embeddings**, **BGE Cross-Encoder**, Top-50 candidates and Top-5 context. Only approved-status filtering is used; no gold event/category filter is provided. The test uses a unique temporary index and removes its own container after completion.

**480 retrievable cases** are evaluated per configuration. The other **20 malformed-OCR cases** must fail the upload integrity gate and cannot honestly be counted as valid indexed-document retrieval tests.

| Configuration | Cases | Recall@5 | Precision@5 | MRR@5 | Mean latency | Errors |
|---|---:|---:|---:|---:|---:|---:|
| dense | 480 | 88.75% | 46.17% | 0.9082 | 24 ms | 0 |
| hybrid | 480 | 96.25% | 50.25% | 0.9220 | 23 ms | 0 |
| hybrid_rerank | 480 | 63.96% | 34.21% | 0.8753 | 2269 ms | 0 |
| full | 480 | 63.96% | 34.21% | 0.8753 | 2363 ms | 0 |

Cross-Encoder reranking with the current dynamic score threshold/gap reduces complete supporting-document coverage on this corpus. The experiment does **not** support claiming that every added stage improves recall. Full-pipeline query reformulation fired on **0** cases: the current gate checks for an empty selected context, not whether every required clause has been retrieved. A confident but incomplete context can therefore miss a rounding/scope document without triggering rewrite.

This is why the upload calculator supplies the entire bounded, reviewed policy set rather than passing that set through shared-index Top-K selection. A future retrieval change should test clause completeness on development data before being evaluated on fresh human-reviewed examples. These CPU timings on 156 documents do not establish million-document scale or sub-50 ms service latency.

## Assertion-level sample audit

A deterministic stratified sample selects **one saved first-run response per family (25 total)**. The judge uses the configured model, so this is not independent human adjudication. Two responses contain no extracted factual assertions, leaving **23 valid faithfulness cases**.

| Judge evidence | Faithfulness | Reference-claim recall | Valid faithfulness cases |
|---|---:|---:|---:|
| Policy documents + order facts | 88.60% | 52.00% | 23 / 25 |
| Policy documents + order facts + trusted Decimal trace | 99.21% | 49.33% | 23 / 25 |

The policy-only audit flagged statements such as “the arithmetic is computed with Decimal” because policy files do not establish implementation provenance. The tool-aware audit includes the actual server calculation trace as evidence. **Both reports are preserved.** Neither sample score is a 500-case faithfulness score, and neither is a retrieval-plus-generation benchmark.

Low reference-claim coverage exposed missing eligibility explanations even when the final price was correct. The new condition checks address the answer format, but their full judge evaluation is blocked by the API balance; improvement is not claimed without measurement.

## Artifacts

- [Complete first agent report, gzip JSON](../evaluation/results/golden-500-agent.json.gz)
- [Incomplete condition-explanation attempt, gzip JSON](../evaluation/results/golden-500-agent-with-conditions.json.gz)
- [Full retrieval traces, gzip JSON](../evaluation/results/golden-480-retrieval.json.gz)
- [Policy-only assertion audit, gzip JSON](../evaluation/results/golden-25-assertions.json.gz)
- [Tool-aware assertion audit, gzip JSON](../evaluation/results/golden-25-assertions-with-tools.json.gz)

The [summary manifest](../evaluation/results/golden-summary.json) records uncompressed report hashes. Reproduction commands and evaluator limitations are in the [dataset guide](../examples/evaluation/golden/README.md).
