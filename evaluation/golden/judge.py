"""Assertion-level audit of a stratified sample of saved real agent answers.

Sampling is explicit; scores are never attributed to all 500 cases. This judge can
be run with --limit 500, but uses several paid API calls per evaluated answer.
"""
import argparse
import asyncio
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

from evaluation.golden.benchmark import DEFAULT_DATA, read_cases, sha
from evaluation.rag_benchmark import AssertionJudge


async def main(args):
    from anthropic import AsyncAnthropic
    from dotenv import load_dotenv
    load_dotenv()
    source=json.loads(Path(args.responses).read_text())
    if source['dataset_sha256']!=sha(args.dataset):raise ValueError('Responses and dataset fingerprints differ')
    cases={case['id']:case for case in read_cases(args.dataset)}
    rows=sorted(source['results'],key=lambda row:(row['id'][-2:],row['family']))[:args.limit]
    output=Path(args.output)
    if output.exists():raise ValueError('Choose a new report path')
    kwargs={'api_key':os.environ['ANTHROPIC_API_KEY'],'timeout':90.0,'max_retries':1}
    if os.getenv('ANTHROPIC_BASE_URL'):kwargs['base_url']=os.environ['ANTHROPIC_BASE_URL']
    model=os.getenv('GOLDEN_JUDGE_MODEL',os.getenv('ANSWER_MODEL',os.getenv('ANTHROPIC_MODEL','claude-sonnet-4-6')))
    semaphore=asyncio.Semaphore(3)
    async with AsyncAnthropic(**kwargs) as client:
        judge=AssertionJudge(client,model)
        async def one(row):
            case=cases[row['id']]
            async with semaphore:
                if 'actual' not in row:
                    return {'id':row['id'],'judge_failed':True,'error':'Missing agent response'}
                evidence_data={'order':case['order'],'policies':case['policies']}
                if args.include_tool_trace and row['actual'].get('calculation'):
                    evidence_data['trusted_tool_execution']={'engine':'Python Decimal', 'result':row['actual']['calculation']}
                evidence=json.dumps(evidence_data,ensure_ascii=False)
                result=await judge.evaluate(case['question'],row['actual']['answer'],evidence,case['reference_claims'])
                if result.get('judge_failed'):result['error']='Judge response unavailable or invalid'
                return {'id':row['id'],'family':case['family'],**result}
        results=await asyncio.gather(*(one(row) for row in rows))
    summary={'sample_size':len(results),'judge_failures':sum(row.get('judge_failed',False) for row in results)}
    for metric in ('faithfulness','answer_claim_recall','retrieved_claim_recall','unsupported_claim_rate'):
        values=[row[metric] for row in results if row.get(metric) is not None]
        summary[metric]=statistics.mean(values) if values else None
        summary[metric+'_valid_cases']=len(values)
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'dataset_sha256':sha(args.dataset),
        'response_report_sha256':sha(args.responses),'judge_model':model,
        'include_tool_trace':args.include_tool_trace,
        'scope':'Stratified answer audit only; reviewed uploads, not retrieved contexts. Retrieved-claim recall here means coverage in supplied upload evidence.',
        'sampling':'One case per family before a second; includes clarification and validation-failure responses.',
        'independence':'Same configured provider/model by default; LLM judgments require human review.',
        'summary':summary,'results':results}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default=str(DEFAULT_DATA))
    parser.add_argument('--responses',default='reports/golden-500-agent.json')
    parser.add_argument('--output',default='reports/golden-25-assertions.json')
    parser.add_argument('--limit',type=int,default=25)
    parser.add_argument('--include-tool-trace',action='store_true',help='Include trusted Decimal execution evidence so tool-provenance claims can be judged')
    args=parser.parse_args()
    if not 1<=args.limit<=500:parser.error('limit must be 1..500')
    asyncio.run(main(args))
