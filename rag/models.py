"""Local BGE embeddings and lazy Cross-Encoder reranking (CPU supported)."""
import asyncio
import os
import threading


class BGEEmbeddings:
    def __init__(self, model_name=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'), device=os.getenv('RAG_DEVICE', 'cpu'))
        self.dimensions = self.model.get_sentence_embedding_dimension()
        self._lock = threading.Lock()

    def count_tokens(self, text):
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def embed_documents(self, texts):
        with self._lock:
            # Reject silent truncation, especially of long formulas.
            if any(self.count_tokens(t) + 2 > self.model.max_seq_length for t in texts):
                raise ValueError('A rule block exceeds the embedding model token limit; split or simplify the source formula')
            return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.embed_documents(['Represent this sentence for searching relevant passages: ' + text])[0]


class CrossEncoderReranker:
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv('RERANK_MODEL', 'BAAI/bge-reranker-base')
        self._model = None
        self._lock = threading.Lock()

    def _rank(self, query, candidates):
        from sentence_transformers import CrossEncoder
        import torch
        with self._lock:
            if self._model is None:
                self._model = CrossEncoder(self.model_name, device=os.getenv('RAG_DEVICE', 'cpu'), max_length=512)
            for row in candidates:
                length = len(self._model.tokenizer.encode(query, row['content'], add_special_tokens=True))
                if length > self._model.max_length:
                    raise ValueError('Question and rule exceed the reranker token limit; shorten the question or reduce RAG_CHUNK_TOKENS')
            scores = self._model.predict([(query, row['content']) for row in candidates], activation_fct=torch.nn.Sigmoid())
        ranked = [{**row, 'rerank_score': float(score), 'score': float(score)} for row, score in zip(candidates, scores)]
        return sorted(ranked, key=lambda row: (-row['rerank_score'], row['chunk_id']))

    async def rank(self, query, candidates):
        return await asyncio.to_thread(self._rank, query, candidates) if candidates else []
