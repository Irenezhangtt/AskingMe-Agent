"""Optional real ES contract test, with deterministic embeddings (no downloads).

ES_TEST_URL=http://localhost:19200 .venv/bin/python -m unittest discover -s tests -p test_elasticsearch_integration.py -v
Uses and removes a unique index; never touches the application rule index.
"""
import os
import unittest
import uuid

from rag.chunking import estimate_tokens
from rag.elasticsearch_store import ElasticsearchKnowledgeBase


class TestEmbeddings:
    dimensions = 3
    count_tokens = staticmethod(estimate_tokens)

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


@unittest.skipUnless(os.getenv('ES_TEST_URL'), 'Set ES_TEST_URL for the optional real ES test')
class ElasticsearchIntegrationTests(unittest.TestCase):
    def test_ingestion_hybrid_filters_versions_and_deletion(self):
        from elasticsearch import Elasticsearch
        client = Elasticsearch(os.environ['ES_TEST_URL'])
        index = 'askingme-test-' + uuid.uuid4().hex
        kb = ElasticsearchKnowledgeBase(client=client, embeddings=TestEmbeddings(), index=index)
        try:
            original = kb.import_document(title='Threshold discount rules', content='# Threshold discount\nAppliances receive CNY 50 off orders of CNY 300 or more.', status='approved',
                source_name='test.md', metadata={'category': 'Appliances', 'effective_from': '2026-01-01', 'effective_to': '2026-12-31'})
            draft = kb.import_document(title='Threshold discount rules', content='# Updated threshold discount\nAppliances receive CNY 60 off orders of CNY 300 or more.', status='draft',
                source_name='test-v2.md', replaces_document_id=original['document_id'], metadata={'category': 'Appliances'})
            rows = kb.retrieve('Appliance threshold discount', filters={'category': 'Appliances', 'as_of': '2026-09-22'})
            self.assertTrue(rows)
            self.assertEqual({r['document_id'] for r in rows}, {original['document_id']})
            self.assertTrue(all('rrf_score' in r for r in rows))
            self.assertEqual(kb.retrieve('Appliances', filters={'category': 'Clothing'}), [])
            self.assertEqual(kb.retrieve('Appliances', filters={'as_of': '2027-01-01'}), [])
            kb.approve_document(draft['document_id'])
            self.assertEqual(kb.get_document(original['document_id'])['status'], 'archived')
            self.assertEqual({r['document_id'] for r in kb.retrieve('Appliances')}, {draft['document_id']})
            kb.delete_document(draft['document_id'])
            self.assertEqual(kb.retrieve('Appliances'), [])
            self.assertEqual(len(kb.list_documents()), 1)
        finally:
            client.indices.delete(index=index, ignore_unavailable=True)
            client.close()
