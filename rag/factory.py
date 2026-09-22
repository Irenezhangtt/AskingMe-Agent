"""Application and offline benchmark share one production RAG factory."""
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from rag.elasticsearch_store import ElasticsearchKnowledgeBase
from rag.graph import PricingRAG
from rag.llm import PricingLLM
from rag.models import CrossEncoderReranker


def build_rag():
    store = ElasticsearchKnowledgeBase()
    if store.doc_count == 0 and os.getenv('RAG_LOAD_EXAMPLES', 'true').lower() == 'true':
        for path in sorted((Path(__file__).resolve().parent.parent / 'examples' / 'pricing').glob('*.md')):
            store.import_document(title=path.read_text().splitlines()[0].lstrip('# ').strip(), content=path.read_text(), status='approved',
                                  source_name=path.name, metadata={'category': 'Appliances', 'event': 'Demo Promotion'})
    kwargs = {'api_key': os.environ['ANTHROPIC_API_KEY']}
    if os.getenv('ANTHROPIC_BASE_URL'):
        kwargs['base_url'] = os.environ['ANTHROPIC_BASE_URL']
    llm = PricingLLM(AsyncAnthropic(**kwargs), os.getenv('ANSWER_MODEL', os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')))
    return PricingRAG(store, CrossEncoderReranker(), llm,
                      max_rewrites=int(os.getenv('RAG_MAX_REWRITES', '1')),
                      candidate_k=int(os.getenv('RAG_CANDIDATE_K', '50')),
                      min_score=float(os.getenv('RAG_MIN_RERANK_SCORE', '0.35')),
                      token_budget=int(os.getenv('RAG_CONTEXT_TOKENS', '2400')))
