"""Real ES/BGE/Cross-Encoder retrieval ablations on the generated golden corpus.

Creates and deletes only a unique test index. Requires a running Elasticsearch URL.
No fake embeddings, no hard event/category filters, no generated answers or LLM judge.
"""
import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluation.golden.benchmark import DEFAULT_DATA, read_cases, sha
from evaluation.rag_benchmark import CONFIGURATIONS, retrieval_metrics


async def main(args):
    from anthropic import AsyncAnthropic
    from dotenv import load_dotenv
    from rag.elasticsearch_store import ElasticsearchKnowledgeBase
    from rag.graph import PricingRAG
    from rag.llm import PricingLLM
    from rag.models import BGEEmbeddings, CrossEncoderReranker
    load_dotenv()
    cases = [case for case in read_cases(args.dataset) if case.get('retrieval_evaluable', True)]
    if args.limit:
        cases = sorted(cases, key=lambda case:(case['id'][-2:],case['family']))[:args.limit]
    corpus_path = Path(args.dataset).with_name('corpus.jsonl')
    corpus = read_cases(corpus_path)
    embeddings = await asyncio.to_thread(BGEEmbeddings)
    index = 'askingme-golden-' + uuid.uuid4().hex[:16]
    store = await asyncio.to_thread(ElasticsearchKnowledgeBase, embeddings=embeddings, index=index)
    kwargs = {'api_key': os.environ['ANTHROPIC_API_KEY'], 'timeout': 60.0, 'max_retries': 1}
    if os.getenv('ANTHROPIC_BASE_URL'):
        kwargs['base_url'] = os.environ['ANTHROPIC_BASE_URL']
    llm = PricingLLM(AsyncAnthropic(**kwargs), os.getenv('ANSWER_MODEL', os.getenv('ANTHROPIC_MODEL','claude-sonnet-4-6')))
    rag = PricingRAG(store, CrossEncoderReranker(), candidate_k=50, max_rewrites=1, min_score=.35, token_budget=2400, llm=llm)
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint = output.with_suffix('.jsonl')
    if checkpoint.exists() or output.exists():
        await llm.client.close()
        store.client.indices.delete(index=index)
        store.client.close()
        raise ValueError('Choose a new output path; existing reports are never overwritten')
    rows = []
    try:
        ids, texts, metadatas = [], [], []
        for doc in corpus:
            doc_id = hashlib.sha256(doc['source_name'].encode()).hexdigest()[:24]
            chunks, metadata = store._prepare_chunks(doc['content'], doc_id, doc['title'],
                {**doc['metadata'], 'source_name':doc['source_name'], 'title':doc['title'], 'status':'approved', 'version':'gold-v1', 'document_id':doc_id})
            for i,(text,meta) in enumerate(zip(chunks,metadata)):
                ids.append(f'{doc_id}:{i}');texts.append(text);metadatas.append({**meta,'chunk_index':i})
        await asyncio.to_thread(store._collection.add, ids, texts, metadatas)
        print(f'Indexed {len(ids)} real BGE chunks from {len(corpus)} documents in isolated ES index',flush=True)
        await rag.run(cases[0]['question'],generate=False,use_rewrite=False)
        for configuration, config in CONFIGURATIONS.items():
            for count,case in enumerate(cases,1):
                started = time.monotonic()
                result = {'id':case['id'],'family':case['family'],'configuration':configuration}
                try:
                    state = await rag.run(case['question'], generate=False, top_k=5, **config)
                    docs = state['documents']
                    result.update(retrieval_metrics([doc['source_name'] for doc in docs],case['relevant_documents']))
                    result.update({'retrieved_documents':[doc['source_name'] for doc in docs], 'trace':state['trace']})
                except Exception as error:
                    result.update({'error':type(error).__name__, 'recall@5':0.0,'precision@5':0.0,'mrr@5':0.0})
                result['latency_ms'] = round((time.monotonic()-started)*1000,2)
                rows.append(result)
                with checkpoint.open('a') as stream:
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                if count%50==0 or count==len(cases):
                    current=[row for row in rows if row['configuration']==configuration]
                    print(f'{configuration}: {count}/{len(cases)}, Recall@5={statistics.mean(row["recall@5"] for row in current):.4f}',flush=True)
        summaries={}
        for name in CONFIGURATIONS:
            group=[row for row in rows if row['configuration']==name]
            summaries[name]={'cases':len(group),'errors':sum('error' in row for row in group),
                **{metric:statistics.mean(row[metric] for row in group) for metric in ('recall@5','precision@5','mrr@5','latency_ms')}}
        report={'timestamp':datetime.now(timezone.utc).isoformat(),'dataset_sha256':sha(args.dataset),'corpus_sha256':sha(corpus_path),
            'dataset_size':500,'evaluated_cases':len(cases),'excluded_integrity_cases':20,
            'exclusion_reason':'Malformed uploaded OCR cases have no valid retrievable source and test the upload integrity gate separately.',
            'embedding_model':os.getenv('EMBEDDING_MODEL','BAAI/bge-small-en-v1.5'),
            'reranker':rag.reranker.model_name,'device':os.getenv('RAG_DEVICE','cpu'),'candidate_k':50,'top_k':5,
            'filters':{'status':'approved'},'cache':'disabled; model warm-up excluded; backend/OS caches may be warm',
            'generated_answers':False,'claim_grounding_evaluated':False,
            'summary':summaries,'results':rows}
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summaries,indent=2))
    finally:
        await llm.client.close()
        try:
            await asyncio.to_thread(store.client.indices.delete,index=index)
        finally:
            await asyncio.to_thread(store.client.close)
        print(f'Removed only test index {index}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default=str(DEFAULT_DATA))
    parser.add_argument('--output',default='reports/golden-retrieval.json')
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('limit must be positive')
    asyncio.run(main(args))
