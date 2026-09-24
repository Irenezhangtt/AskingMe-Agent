"""Validate synthetic gold or evaluate real upload-price model responses.

python -m evaluation.golden.benchmark --mode validate
python -m evaluation.golden.benchmark --mode agent --output reports/golden-agent.json
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from rag.calculator import calculate_final_price
from rag.chunking import RuleChunker
from rag.policy_uploads import UploadedPriceRequest, calculate_uploaded_price
from evaluation.golden.generate import generate

class EvaluationBlocked(Exception):
    pass


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / 'examples/evaluation/golden/cases.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_cases(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def validate(path):
    cases = read_cases(path)
    if len(cases) != 500 or len({case['id'] for case in cases}) != 500:
        raise ValueError('Expected exactly 500 unique cases')
    if len({case['question'] for case in cases}) != 500:
        raise ValueError('Duplicate questions')
    if cases != generate()[0]:
        raise ValueError('Dataset differs from the versioned generator')
    corpus = {doc['source_name']: doc for doc in read_cases(Path(path).with_name('corpus.jsonl'))}
    checked_prices = 0
    groups = {}
    for case in cases:
        request = UploadedPriceRequest.model_validate({'policies': case['policies'], 'order': case['order']})
        if not set(case['relevant_documents']) <= corpus.keys():
            raise ValueError('Dangling source label')
        groups.setdefault(case['family'], set()).add(case['split'])
        if case['expected']['type'] == 'calculation':
            actual = calculate_final_price(case['expected']['expression'])['final_price']
            if actual != case['expected']['final_price']:
                raise ValueError(f'Independent arithmetic disagreement in {case["id"]}')
            checked_prices += 1
        for policy in request.policies:
            try:
                RuleChunker(32).split(policy.content, parent_id='validation')
            except ValueError:
                if case['family'] != 'malformed_parentheses':
                    raise
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError('A rule family leaks across data splits')
    for doc in corpus.values():
        RuleChunker(32).split(doc['content'], parent_id='corpus-validation')
    return {'dataset_size': len(cases), 'unique_questions': 500, 'corpus_documents': len(corpus),
        'independent_price_crosschecks': checked_prices, 'clarification_cases': 500-checked_prices,
        'families': dict(Counter(case['family'] for case in cases)), 'dataset_sha256': sha(path),
        'notice': 'Fixture/label validation only. This is not an agent accuracy measurement.'}


def score(case, result):
    expected = case['expected']
    model_invalid = bool(result.get('validation_failed'))
    outcome = result.get('type') == expected['type'] and not model_invalid
    price = result.get('calculation', {}).get('final_price')
    currency = result.get('calculation', {}).get('currency')
    exact = (price == expected['final_price'] and currency == expected['currency']) if expected['type'] == 'calculation' else None
    # Answer references, not merely sources passed into the model, determine citation coverage.
    sources = {row['source_id']: row['source_name'] for row in result.get('sources', [])}
    cited = {sources[int(index)] for index in re.findall(r'\[(\d+)\]', result.get('answer', '')) if int(index) in sources}
    required = set(expected['required_source_names'])
    citation_recall = len(required & cited)/len(required) if required else None
    excluded = {row['source_name'] for row in result.get('excluded', [])}
    excludes_correct = excluded == set(expected['excluded_source_names'])
    false_price = expected['type'] == 'clarification' and bool(result.get('calculation'))
    numeric_correct = exact if exact is not None else not false_price
    return {'outcome_correct': outcome, 'price_exact': exact, 'citation_recall': citation_recall,
        'excluded_correct': excludes_correct, 'false_price': false_price,
        'validation_failed': model_invalid, 'cited_sources': sorted(cited),
        'pass': bool(outcome and numeric_correct and excludes_correct),
        'fully_cited_pass': bool(outcome and numeric_correct and excludes_correct and (citation_recall is None or citation_recall == 1))}


def provider_blocker(error):
    body = getattr(error, 'body', None)
    if isinstance(body, dict):
        detail = body.get('error', {})
        message = detail.get('message', '') if isinstance(detail, dict) else ''
        if 'credit balance' in message.lower() and 'too low' in message.lower():
            return 'credit_balance_exhausted'
    return None


def summarize(rows):
    def ratio(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return sum(values)/len(values) if values else None
    calculations = [row for row in rows if row['expected_type'] == 'calculation']
    clarifications = [row for row in rows if row['expected_type'] == 'clarification']
    latencies = sorted(row['latency_ms'] for row in rows)
    return {'cases': len(rows), 'completed_responses': sum('actual' in row for row in rows),
        'not_run': sum(bool(row.get('not_run')) for row in rows),
        'provider_errors': sum('error' in row for row in rows),
        'pass_rate': sum(bool(row.get('pass')) for row in rows)/len(rows) if rows else None,
        'fully_cited_pass_rate': sum(bool(row.get('fully_cited_pass')) for row in rows)/len(rows) if rows else None,
        'price_exact_rate': sum(bool(row.get('price_exact')) for row in calculations)/len(calculations) if calculations else None,
        'calculation_cases': len(calculations), 'clarification_cases': len(clarifications),
        'clarification_outcome_rate': sum(bool(row.get('outcome_correct')) for row in clarifications)/len(clarifications) if clarifications else None,
        'false_price_count': sum(bool(row.get('false_price')) for row in rows),
        'mean_citation_recall_valid_responses': ratio('citation_recall'),
        'p50_ms': statistics.median(latencies) if latencies else None,
        'p95_ms': latencies[max(0, (95*len(latencies)+99)//100-1)] if latencies else None}


async def evaluate(args):
    from dotenv import load_dotenv
    from anthropic import AsyncAnthropic
    from rag.llm import PricingLLM
    load_dotenv()
    cases = read_cases(args.dataset)
    if args.split != 'all':
        cases = [case for case in cases if case['split'] == args.split]
    if args.limit:
        # Deterministic stratified prefix: one per family before taking a second.
        cases = sorted(cases, key=lambda case: (case['id'][-2:], case['family']))[:args.limit]
    model = os.getenv('ANSWER_MODEL', os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6'))
    fingerprint = {'dataset_sha256': sha(args.dataset), 'model': model, 'split': args.split,
        'case_ids': [case['id'] for case in cases],
        'implementation_sha256': hashlib.sha256(b''.join((ROOT/name).read_bytes() for name in
            ('rag/policy_uploads.py','rag/llm.py','rag/calculator.py','rag/chunking.py','rag/rule_integrity.py'))).hexdigest()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix('.jsonl')
    metadata = output.with_suffix('.meta.json')
    completed = {}
    if args.resume and metadata.exists():
        if json.loads(metadata.read_text()) != fingerprint:
            raise ValueError('Resume fingerprint differs; use a new output path')
        for row in read_cases(checkpoint) if checkpoint.exists() else []:
            completed[row['id']] = row
    elif checkpoint.exists() or metadata.exists():
        raise ValueError('Output already exists. Use --resume or choose a new output path')
    if args.retry_errors:
        if output.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            shutil.copyfile(output, output.with_name(output.stem + '-before-retry-' + stamp + '.json'))
        completed = {key: row for key, row in completed.items() if 'error' not in row and not row.get('not_run')}
    metadata.write_text(json.dumps(fingerprint, indent=2)+'\n')
    kwargs = {'api_key': os.environ['ANTHROPIC_API_KEY'], 'timeout': 90.0, 'max_retries': 2}
    if os.getenv('ANTHROPIC_BASE_URL'):
        kwargs['base_url'] = os.environ['ANTHROPIC_BASE_URL']
    semaphore = asyncio.Semaphore(args.concurrency)
    stopped_for_credit = asyncio.Event()
    async with AsyncAnthropic(**kwargs) as client:
        async def guarded_create(**kwargs):
            if stopped_for_credit.is_set():
                raise EvaluationBlocked('credit_balance_exhausted')
            return await client.messages.create(**kwargs)
        llm = PricingLLM(SimpleNamespace(messages=SimpleNamespace(create=guarded_create)), model)
        async def one(case):
            if case['id'] in completed:
                return
            async with semaphore:
                started = time.monotonic()
                row = {'id': case['id'], 'family': case['family'], 'split': case['split'], 'expected_type': case['expected']['type']}
                try:
                    request = UploadedPriceRequest.model_validate({'policies': case['policies'], 'order': case['order']})
                    result = await asyncio.wait_for(calculate_uploaded_price(llm, request), timeout=100)
                    row.update(score(case, result))
                    row.update({'actual': result, 'expected': case['expected']})
                except EvaluationBlocked:
                    row.update({'not_run': True, 'blocked_reason': 'credit_balance_exhausted', 'pass': False, 'fully_cited_pass': False, 'outcome_correct': False})
                except Exception as error:
                    if provider_blocker(error):
                        stopped_for_credit.set()
                        row['blocked_reason'] = provider_blocker(error)
                    row.update({'error': type(error).__name__, 'pass': False, 'fully_cited_pass': False,
                                'price_exact': False if case['expected']['type'] == 'calculation' else None,
                                'outcome_correct': False})
                row['latency_ms'] = round((time.monotonic()-started)*1000, 2)
                completed[case['id']] = row
                with checkpoint.open('a') as stream:
                    stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                if len(completed)%25 == 0:
                    print(f'Completed {len(completed)}/{len(cases)}; failures={sum(not r["pass"] for r in completed.values())}', flush=True)
        await asyncio.gather(*(one(case) for case in cases))
    rows = [completed[case['id']] for case in cases]
    report = {'timestamp': datetime.now(timezone.utc).isoformat(), **fingerprint,
        'provenance': 'Synthetic fixture evaluation; not human-annotated production accuracy.',
        'scope': 'Reviewed upload calculation path; no retrieval or OCR in this run. Clarification outcome is scored, not semantic reason completeness.',
        'summary': summarize(rows),
        'by_family': {family: summarize([row for row in rows if row['family']==family]) for family in sorted({row['family'] for row in rows})},
        'by_split': {split: summarize([row for row in rows if row['split']==split]) for split in sorted({row['split'] for row in rows})},
        'results': rows}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['validate','agent'], default='validate')
    parser.add_argument('--dataset', default=str(DEFAULT_DATA))
    parser.add_argument('--output', default='reports/golden-agent.json')
    parser.add_argument('--split', choices=['all','development','evaluation'], default='all')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true', help='With --resume, retry only provider errors/not-run cases after restoring service; preserve the earlier report')
    args = parser.parse_args()
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    if not 1 <= args.concurrency <= 12 or (args.limit is not None and args.limit < 1):
        parser.error('Use concurrency 1..12 and a positive limit')
    if args.mode == 'validate':
        print(json.dumps(validate(args.dataset), indent=2))
    else:
        asyncio.run(evaluate(args))


if __name__ == '__main__':
    main()
