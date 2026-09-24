"""Export immutable synthetic evaluation reports and a concise measured-results page."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'evaluation/results'
NAMES=['golden-500-agent','golden-500-agent-with-conditions','golden-480-retrieval',
       'golden-25-assertions','golden-25-assertions-with-tools']


def percent(value):
    return f'{value*100:.2f}%' if value is not None else 'Not measured'


def main():
    reports={name:json.loads((ROOT/f'reports/{name}.json').read_text()) for name in NAMES}
    DEST.mkdir(parents=True,exist_ok=True)
    archives={}
    for name in NAMES:
        content=(ROOT/f'reports/{name}.json').read_bytes()
        destination=DEST/f'{name}.json.gz'
        destination.write_bytes(gzip.compress(content,mtime=0))
        archives[name]={'file':destination.name,'uncompressed_sha256':hashlib.sha256(content).hexdigest()}
    summary={'dataset_sha256':reports[NAMES[0]]['dataset_sha256'], 'archives':archives,
        'agent_initial':reports[NAMES[0]]['summary'], 'agent_with_conditions':reports[NAMES[1]]['summary'],
        'retrieval':reports[NAMES[2]]['summary'], 'assertions_policy_only':reports[NAMES[3]]['summary'],
        'assertions_with_tool_evidence':reports[NAMES[4]]['summary'],
        'provider_blocker':'A follow-up minimal request confirmed insufficient Anthropic API credit after the condition-explanation run began failing.',
        'limits':'Synthetic, English, inspected during development; no production/human-label accuracy claim. Current condition-explanation version is incompletely model-validated.'}
    (DEST/'golden-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    initial=reports[NAMES[0]]
    enhanced=reports[NAMES[1]]
    retrieval=reports[NAMES[2]]
    judge=reports[NAMES[4]]['summary']
    failed=[{'id':row['id'],'family':row['family'],'expected':row['expected'],
             'actual':{key:row['actual'][key] for key in ('type','answer','validation_failed') if key in row['actual']}}
             for row in initial['results'] if not row['pass']]
    (DEST/'initial-agent-failures.json').write_text(json.dumps(failed,indent=2)+'\n')
    retrieval_rows='\n'.join(f'| {name} | {data["cases"]} | {percent(data["recall@5"])} | {percent(data["precision@5"])} | {data["mrr@5"]:.4f} | {data["latency_ms"]:.0f} ms | {data["errors"]} |' for name,data in retrieval['summary'].items())
    failure_rows='\n'.join(f'| `{row["id"]}` | {row["family"]} | Unvalidated model plan; no computed price |' for row in failed)
    rewrite_calls=sum(any(stage.get('stage')=='query_reformulation' for stage in row.get('trace',[])) for row in retrieval['results'] if row['configuration']=='full')
    text=f'''# Measured golden-set evaluation

Evaluation date: **2026-09-22**. All metrics below are measured on the versioned **synthetic** dataset, not copied from the resume brief. Human annotation and production validation remain pending.

**[500 cases and corpus](../examples/evaluation/golden/)** · **[Machine-readable summary](../evaluation/results/golden-summary.json)** · **[Resume implementation audit](resume-implementation-audit.md)**

## Dataset and protocol

- **500 unique questions**, **25 families × 20 cases**, **156 fictional English policy documents**.
- **390 numerical targets**, computed with an independent rational/integer-cent oracle; **110 clarification targets**.
- Family-separated split: **100 development / 400 evaluation**. The synthetic cases have been inspected during development; the second split is not independent human-held-out evidence.
- Dataset SHA-256: `{initial['dataset_sha256']}`.
- Agent model alias: `{initial['model']}`. Provider-default sampling; no deterministic-generation claim.
- The agent benchmark uses reviewed upload text, explicit order facts and the real pricing planner/Decimal tool. It does not measure OCR or retrieval. Fifty cases terminate in local validity/integrity guards; the rest call the model.

## Complete first agent run

| Measure | Observed result |
|---|---:|
| Requests evaluated | 500 / 500 |
| Expected outcome and price/exclusions satisfied | **492 / 500 ({percent(initial['summary']['pass_rate'])})** |
| Exact currency and amount on numeric cases | **382 / 390 ({percent(initial['summary']['price_exact_rate'])})** |
| Expected clarification outcome | **110 / 110** |
| Fully cited expected outcomes | **492 / 500** |
| Provider errors | **0** |
| p50 / p95 request latency | **{initial['summary']['p50_ms']:.0f} / {initial['summary']['p95_ms']:.0f} ms** |

The eight failures returned a validation fallback rather than a verified amount. This run did not emit a structured price on any expected-clarification case. Clarification-action scoring does not verify the full semantic quality of the question asked.

| Failed case | Family | Recorded outcome |
|---|---|---|
{failure_rows}

The original raw failed plans were not retained, so schema, syntax and truncation causes cannot be separated retrospectively. Three exploratory retries succeeded; they **do not replace** the original first-pass results. [Failure records](../evaluation/results/initial-agent-failures.json).

## Current condition-explanation version: incomplete model validation

The first answer format often gave arithmetic without explaining exception scope. The current version requires cited `rule_checks` covering eligibility, exclusions, grouping and rounding. It also emits a safe validation-error category for future debugging.

A second 500-case attempt recorded **{enhanced['summary']['completed_responses']} completed responses** and **{enhanced['summary']['provider_errors']} provider errors**. A follow-up minimal request confirmed **insufficient Anthropic API credit**. This incomplete attempt is retained; its error-inclusive pass rate is **not** an estimate of model capability or evidence that the remaining cases passed.

The two model-response validation failures before that service interruption reported `CalculationError` in repeated-threshold cases. The remaining unverified requests must be rerun after service is restored. The current version therefore must not be advertised as having the first run's complete 500-case score.

```bash
# After restoring API credit, retry only service-error/not-run cases under the same code/model/data.
.venv/bin/python -m evaluation.golden.benchmark --mode agent \\
  --output reports/golden-500-agent-with-conditions.json \\
  --resume --retry-errors --concurrency 6
```

The retry runner preserves prior reports/checkpoint attempts and stops further paid requests on a recognized credit-exhaustion response, while still evaluating local guards.

## Real retrieval ablations

Actual **Elasticsearch 8.17.2**, **BGE small English embeddings**, **BGE Cross-Encoder**, Top-50 candidates and Top-5 context. Only approved-status filtering is used; no gold event/category filter is provided. The test uses a unique temporary index and removes its own container after completion.

**480 retrievable cases** are evaluated per configuration. The other **20 malformed-OCR cases** must fail the upload integrity gate and cannot honestly be counted as valid indexed-document retrieval tests.

| Configuration | Cases | Recall@5 | Precision@5 | MRR@5 | Mean latency | Errors |
|---|---:|---:|---:|---:|---:|---:|
{retrieval_rows}

Cross-Encoder reranking with the current dynamic score threshold/gap reduces complete supporting-document coverage on this corpus. The experiment does **not** support claiming that every added stage improves recall. Full-pipeline query reformulation fired on **{rewrite_calls}** cases: the current gate checks for an empty selected context, not whether every required clause has been retrieved. A confident but incomplete context can therefore miss a rounding/scope document without triggering rewrite.

This is why the upload calculator supplies the entire bounded, reviewed policy set rather than passing that set through shared-index Top-K selection. A future retrieval change should test clause completeness on development data before being evaluated on fresh human-reviewed examples. These CPU timings on 156 documents do not establish million-document scale or sub-50 ms service latency.

## Assertion-level sample audit

A deterministic stratified sample selects **one saved first-run response per family (25 total)**. The judge uses the configured model, so this is not independent human adjudication. Two responses contain no extracted factual assertions, leaving **{judge['faithfulness_valid_cases']} valid faithfulness cases**.

| Judge evidence | Faithfulness | Reference-claim recall | Valid faithfulness cases |
|---|---:|---:|---:|
| Policy documents + order facts | {percent(reports[NAMES[3]]['summary']['faithfulness'])} | {percent(reports[NAMES[3]]['summary']['answer_claim_recall'])} | {reports[NAMES[3]]['summary']['faithfulness_valid_cases']} / 25 |
| Policy documents + order facts + trusted Decimal trace | {percent(judge['faithfulness'])} | {percent(judge['answer_claim_recall'])} | {judge['faithfulness_valid_cases']} / 25 |

The policy-only audit flagged statements such as “the arithmetic is computed with Decimal” because policy files do not establish implementation provenance. The tool-aware audit includes the actual server calculation trace as evidence. **Both reports are preserved.** Neither sample score is a 500-case faithfulness score, and neither is a retrieval-plus-generation benchmark.

Low reference-claim coverage exposed missing eligibility explanations even when the final price was correct. The new condition checks address the answer format, but their full judge evaluation is blocked by the API balance; improvement is not claimed without measurement.

## Artifacts

- [Complete first agent report, gzip JSON](../evaluation/results/golden-500-agent.json.gz)
- [Incomplete condition-explanation attempt, gzip JSON](../evaluation/results/golden-500-agent-with-conditions.json.gz)
- [Full retrieval traces, gzip JSON](../evaluation/results/golden-480-retrieval.json.gz)
- [Policy-only assertion audit, gzip JSON](../evaluation/results/golden-25-assertions.json.gz)
- [Tool-aware assertion audit, gzip JSON](../evaluation/results/golden-25-assertions-with-tools.json.gz)

The [summary manifest](../evaluation/results/golden-summary.json) records uncompressed report hashes. Reproduction commands and evaluator limitations are in the [dataset guide](../examples/evaluation/golden/README.md).
'''
    (ROOT/'docs/golden-evaluation.md').write_text(text)
    print('Exported measured reports and docs/golden-evaluation.md')


if __name__=='__main__':main()
