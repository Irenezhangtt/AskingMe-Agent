"""Bounded LangGraph retrieve–rerank–rewrite–generate workflow."""
import asyncio
import re
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from rag.chunking import estimate_tokens


class RAGState(TypedDict, total=False):
    question: str
    query: str
    filters: dict
    keywords: list[str]
    candidates: list[dict]
    documents: list[dict]
    attempts: int
    sufficient: bool
    answer: str
    trace: list[dict]
    history: str
    generate: bool
    mode: str
    use_rerank: bool
    use_rewrite: bool
    top_k: int
    token_callback: Any


def select_context(ranked, *, top_k=5, min_score=0.35, score_gap=0.25, token_budget=2400):
    selected, used = [], 0
    if not ranked:
        return selected
    best = ranked[0]['score']
    for row in ranked[:top_k]:
        if row['score'] < min_score or best - row['score'] > score_gap:
            break
        # Budget uses conservative token estimate, and always keeps chunks whole.
        tokens = max(row.get('token_count', 0), estimate_tokens(row['content'])) + estimate_tokens(row.get('heading_path', '')) + 40
        if used + tokens > token_budget:
            continue
        selected.append(row)
        used += tokens
    return selected


class PricingRAG:
    def __init__(self, store, reranker, llm, *, max_rewrites=1, candidate_k=50, min_score=0.35, token_budget=2400):
        if not 0 <= max_rewrites <= 3 or not 1 <= candidate_k <= 200 or not 0 <= min_score <= 1 or token_budget < 1:
            raise ValueError('Invalid RAG configuration')
        self.store, self.reranker, self.llm = store, reranker, llm
        self.max_rewrites, self.candidate_k = max_rewrites, candidate_k
        self.min_score, self.token_budget = min_score, token_budget
        builder = StateGraph(RAGState)
        for name, node in [('query_analysis', self.analyze), ('hybrid_retrieval', self.retrieve),
                           ('rerank', self.rerank), ('query_reformulation', self.rewrite),
                           ('response_generation', self.generate)]:
            builder.add_node(name, node)
        builder.add_edge(START, 'query_analysis')
        builder.add_edge('query_analysis', 'hybrid_retrieval')
        builder.add_edge('hybrid_retrieval', 'rerank')
        builder.add_conditional_edges('rerank', self.route,
                                      {'rewrite': 'query_reformulation', 'generate': 'response_generation', 'end': END})
        builder.add_edge('query_reformulation', 'query_analysis')
        builder.add_edge('response_generation', END)
        self.graph = builder.compile()

    @staticmethod
    def traced(state, stage, started, **detail):
        return state.get('trace', []) + [{'stage': stage, 'latency_ms': round((time.monotonic() - started) * 1000, 2), **detail}]

    async def analyze(self, state):
        started = time.monotonic()
        # Keep numbers, product names and original question intact. Do not infer hard filters.
        query = state.get('query') or state['question']
        keywords = re.findall(r'\d+(?:\.\d+)?|[a-zA-Z][\w-]*|[\u4e00-\u9fff]{2,}', query)
        return {'query': query, 'keywords': keywords,
                'trace': self.traced(state, 'query_analysis', started, query=query, keywords=keywords)}

    async def retrieve(self, state):
        started = time.monotonic()
        rows = await asyncio.to_thread(self.store.retrieve, state['query'], self.candidate_k,
                                       state.get('filters'), state.get('mode', 'hybrid'))
        return {'candidates': rows, 'trace': self.traced(state, 'hybrid_retrieval', started, count=len(rows))}

    async def rerank(self, state):
        started = time.monotonic()
        enabled = state.get('use_rerank', True)
        rows = await self.reranker.rank(state['question'], state['candidates']) if enabled else state['candidates']
        docs = select_context(rows, top_k=state.get('top_k', 5), min_score=self.min_score if enabled else float('-inf'),
                              score_gap=0.25 if enabled else float('inf'), token_budget=self.token_budget)
        return {'documents': docs, 'sufficient': bool(docs),
                'trace': self.traced(state, 'rerank', started, enabled=enabled, selected=len(docs),
                                     top_score=rows[0]['score'] if rows else None)}

    def route(self, state):
        if not state['sufficient'] and state.get('use_rewrite', True) and state.get('attempts', 0) < self.max_rewrites:
            return 'rewrite'
        return 'generate' if state.get('generate', True) else 'end'

    async def rewrite(self, state):
        started = time.monotonic()
        query = await self.llm.rewrite(state['question'], state.get('history', ''))
        # Preserve numerical constraints; the rewrite cannot silently drop or change them.
        if not query.strip() or set(re.findall(r'\d+(?:\.\d+)?', state['question'])) != set(re.findall(r'\d+(?:\.\d+)?', query)):
            query = state['question']
        return {'query': query, 'attempts': state.get('attempts', 0) + 1,
                'trace': self.traced(state, 'query_reformulation', started, query=query)}

    async def generate(self, state):
        started = time.monotonic()
        if not state['documents']:
            answer = 'I could not find sufficiently reliable pricing rules. Please provide the promotion, product category, and applicable date, or confirm with the rule owner. I cannot calculate the final price from the available evidence.'
            if state.get('token_callback'):
                await state['token_callback'](answer)
        else:
            answer = await self.llm.answer(state['question'], state['documents'], state.get('history', ''), state.get('token_callback'))
        return {'answer': answer, 'trace': self.traced(state, 'response_generation', started)}

    async def run(self, question, *, filters=None, history='', generate=True, mode='hybrid',
                  use_rerank=True, use_rewrite=True, top_k=5, token_callback=None):
        if not question.strip() or not 1 <= top_k <= 20:
            raise ValueError('A question and top_k between 1 and 20 are required')
        return await self.graph.ainvoke({'question': question, 'query': question, 'filters': filters or {},
                                        'history': history, 'generate': generate, 'mode': mode,
                                        'use_rerank': use_rerank, 'use_rewrite': use_rewrite,
                                        'top_k': top_k, 'attempts': 0, 'trace': [], 'token_callback': token_callback},
                                       config={'recursion_limit': 24})
