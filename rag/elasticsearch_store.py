"""Elasticsearch 8.x storage, prefiltered BM25/kNN, client-side RRF."""
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from mcp.knowledge_base import KnowledgeBase
from rag.chunking import RuleChunker


def reciprocal_rank_fusion(*rankings, rank_constant=60, limit=50):
    if rank_constant < 1:
        raise ValueError('rank_constant must be positive')
    scores, rows = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, row in enumerate(ranking, 1):
            key = row['chunk_id']
            if key in seen:
                continue
            seen.add(key)
            rows.setdefault(key, row)
            scores[key] = scores.get(key, 0.0) + 1 / (rank_constant + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [{**rows[key], 'score': scores[key], 'rrf_score': scores[key]} for key in ordered]


def filter_clauses(filters=None):
    clauses = [{'term': {'status': 'approved'}}]
    for field, value in (filters or {}).items():
        if field not in {'category', 'event', 'source_name', 'document_id', 'as_of'}:
            raise ValueError(f'Unsupported retrieval filter: {field}')
        if field == 'as_of':
            for date_field, op in [('effective_from', 'lte'), ('effective_to', 'gte')]:
                clauses.append({'bool': {'should': [
                    {'range': {date_field: {op: value}}},
                    {'bool': {'must_not': {'exists': {'field': date_field}}}},
                ], 'minimum_should_match': 1}})
        else:
            clauses.append({'term': {field: value}})
    return clauses


class ESCollection:
    """Lifecycle adapter used by the existing document administration API."""
    def __init__(self, client, index, embeddings):
        self.client, self.index, self.embeddings = client, index, embeddings

    def get(self, where=None, include=None):
        from elasticsearch.helpers import scan
        query = {'bool': {'filter': [{'term': {k: v}} for k, v in where.items()]}} if where else {'match_all': {}}
        rows = list(scan(self.client, index=self.index, query={'query': query, '_source': {'excludes': ['embedding']}}))
        return {'ids': [r['_id'] for r in rows], 'documents': [r['_source']['content'] for r in rows],
                'metadatas': [{k: v for k, v in r['_source'].items() if k != 'content'} for r in rows]}

    def add(self, ids, documents, metadatas):
        from elasticsearch.helpers import bulk
        vectors = self.embeddings.embed_documents(documents)
        actions = [{'_index': self.index, '_id': key, '_source': {**meta, 'content': content, 'embedding': vector}}
                   for key, content, meta, vector in zip(ids, documents, metadatas, vectors)]
        try:
            bulk(self.client, actions, refresh='wait_for')
        except Exception:
            # Remove partial writes before allowing approval/retry.
            self.delete(ids)
            raise

    def update(self, ids, metadatas):
        from elasticsearch.helpers import bulk
        bulk(self.client, [{'_op_type': 'update', '_index': self.index, '_id': key, 'doc': meta}
                           for key, meta in zip(ids, metadatas)], refresh='wait_for')

    def delete(self, ids):
        from elasticsearch.helpers import bulk
        bulk(self.client, [{'_op_type': 'delete', '_index': self.index, '_id': key} for key in ids],
             refresh='wait_for', ignore_status=(404,))

    def count(self):
        return self.client.count(index=self.index)['count']


class ElasticsearchKnowledgeBase(KnowledgeBase):
    def __init__(self, client=None, embeddings=None, index=None):
        from elasticsearch import Elasticsearch
        from rag.models import BGEEmbeddings
        self.client = client or Elasticsearch(os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200'),
                                             api_key=os.getenv('ELASTICSEARCH_API_KEY') or None, request_timeout=30)
        self.embeddings = embeddings or BGEEmbeddings()
        self.index = index or os.getenv('ELASTICSEARCH_INDEX', 'askingme-pricing-en-v1')
        self.chunker = RuleChunker(int(os.getenv('RAG_CHUNK_TOKENS', '400')), self.embeddings.count_tokens)
        self._collection = ESCollection(self.client, self.index, self.embeddings)
        if not self.client.indices.exists(index=self.index):
            properties = {k: {'type': 'keyword'} for k in (
                'document_id', 'parent_id', 'policy_key', 'version', 'status', 'content_hash', 'source_name',
                'replaces_document_id', 'category', 'event', 'formula_type', 'heading_path', 'imported_at', 'approved_at')}
            properties.update({
                'content': {'type': 'text', 'analyzer': 'english'}, 'title': {'type': 'text', 'analyzer': 'english'},
                'summary': {'type': 'text', 'analyzer': 'english'},
                'embedding': {'type': 'dense_vector', 'dims': self.embeddings.dimensions, 'index': True, 'similarity': 'cosine'},
                **{k: {'type': 'date'} for k in ('effective_from', 'effective_to')},
            })
            self.client.indices.create(index=self.index, mappings={'properties': properties})

    def _prepare_chunks(self, content, document_id, title, metadata):
        docs = self.chunker.split(content, parent_id=document_id, title=title, metadata=metadata)
        return [d.page_content for d in docs], [d.metadata for d in docs]

    def _hits(self, response):
        return [{**hit['_source'], 'chunk_id': hit['_id'], 'score': hit['_score'],
                 'chunk': hit['_source'].get('chunk_index', 0)} for hit in response['hits']['hits']]

    def retrieve(self, query, top_k=50, filters=None, mode='hybrid'):
        if not 1 <= top_k <= 200:
            raise ValueError('top_k must be between 1 and 200')
        if mode not in {'dense', 'bm25', 'hybrid'}:
            raise ValueError('Unsupported retrieval mode')
        clauses = filter_clauses(filters)
        common = {'index': self.index, 'size': top_k, 'source_excludes': ['embedding']}

        def lexical():
            return self._hits(self.client.search(**common, query={'bool': {
                'must': {'multi_match': {'query': query, 'fields': ['content', 'title^2', 'summary']}}, 'filter': clauses}}))

        def dense():
            vector = self.embeddings.embed_query(query)
            return self._hits(self.client.search(**common, knn={
                'field': 'embedding', 'query_vector': vector, 'k': top_k,
                'num_candidates': min(10000, top_k * 10), 'filter': {'bool': {'filter': clauses}}}))

        if mode == 'bm25':
            return lexical()
        if mode == 'dense':
            return dense()
        with ThreadPoolExecutor(max_workers=2) as pool:
            lex, vec = pool.submit(lexical), pool.submit(dense)
            return reciprocal_rank_fusion(lex.result(), vec.result(), limit=top_k)

    def search(self, query, top_k=5):
        return self.retrieve(query, top_k=top_k)

    async def search_handler(self, params, context):
        return await asyncio.to_thread(self.retrieve, params['query'], params.get('top_k', 50), params.get('filters'))
