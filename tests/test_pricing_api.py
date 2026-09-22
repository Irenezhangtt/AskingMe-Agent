"""API integration tests with actual FastAPI routes and isolated service fakes."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from api import main
from rag.elasticsearch_store import ElasticsearchKnowledgeBase
from rag.chunking import RuleChunker
from test_knowledge_lifecycle import FakeCollection


class PricingAPITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.memory = SimpleNamespace(get_context=AsyncMock(return_value=SimpleNamespace(to_prompt_text=lambda: 'prior question')),
                                      add_message=AsyncMock())
        self.state = {'answer': 'The final price is CNY 297. [1]', 'sufficient': True,
                      'documents': [{'chunk_id': 'doc:0', 'document_id': 'doc', 'title': 'Discount', 'source_name': 'rules.md', 'version': '1.0', 'score': .9}],
                      'trace': [{'stage': 'rerank', 'latency_ms': 2.1}]}
        self.rag = SimpleNamespace(run=AsyncMock(return_value=self.state))
        main.app.dependency_overrides[main.enforce_demo_quota] = lambda: None
        main.app.dependency_overrides[main.require_admin_key] = lambda: None
        self.patcher = patch.multiple(main, _memory=self.memory, _orchestrator=object(), _rag=self.rag)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        main.app.dependency_overrides.clear()

    def test_chat_uses_graph_and_returns_sources_and_filters(self):
        response = self.client.post('/chat', json={'message': 'What is the final price for CNY 400?', 'filters': {'category': 'Appliances', 'as_of': '2026-09-22'}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['sources'][0]['chunk_id'], 'doc:0')
        self.assertEqual(self.rag.run.call_args.kwargs['filters'], {'category': 'Appliances', 'as_of': '2026-09-22'})
        self.assertEqual(self.memory.add_message.await_count, 2)

    def test_stream_delivers_answer_metadata_and_done(self):
        async def run(*args, **kwargs):
            await kwargs['token_callback'](self.state['answer'])
            return self.state
        self.rag.run.side_effect = run
        response = self.client.post('/chat/stream', json={'message': 'Final price'})
        events = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(''.join(e['delta'] for e in events if e['type'] == 'answer'), self.state['answer'])
        self.assertTrue(any(e['type'] == 'meta' and e['data']['sources'] for e in events))
        self.assertEqual(events[-1]['type'], 'done')

    def test_search_shares_graph_without_generation(self):
        response = self.client.post('/search', params={'query': 'Final price', 'category': 'Appliances'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(self.rag.run.call_args.kwargs['generate'])
        self.assertEqual(self.client.post('/search', params={'query': 'q', 'top_k': 0}).status_code, 422)

    def test_client_cannot_request_drafts_or_bad_dates(self):
        for filters in ({'status': 'draft'}, {'as_of': 'yesterday'}):
            response = self.client.post('/chat', json={'message': 'q', 'filters': filters})
            self.assertEqual(response.status_code, 422)

    def test_administration_preserves_semantic_metadata_and_lifecycle(self):
        kb = ElasticsearchKnowledgeBase.__new__(ElasticsearchKnowledgeBase)
        kb.chunker = RuleChunker()
        kb._collection = FakeCollection()
        manager = SimpleNamespace(_tools={'knowledge_search': SimpleNamespace(handler=kb.search_handler)})
        with patch.object(main, '_tool_manager', manager), patch.object(main, '_invalidate_answer_cache'):
            upload = self.client.post('/knowledge/upload', data={'category': 'Appliances', 'event': 'Demo Promotion', 'effective_from': '2026-11-01'},
                                      files={'file': ('pricing.md', '# Threshold discount\nCNY 50 off orders of CNY 300 or more.'.encode(), 'text/markdown')})
            self.assertEqual(upload.status_code, 200, upload.text)
            doc_id = upload.json()['documents'][0]['document_id']
            row = next(iter(kb._collection.rows.values()))
            self.assertEqual(row['metadata']['category'], 'Appliances')
            self.assertEqual(row['metadata']['heading_path'], 'Threshold discount')
            self.assertEqual(row['metadata']['status'], 'draft')
            approved = self.client.post(f'/knowledge/documents/{doc_id}/approve')
            self.assertEqual(approved.json()['document']['status'], 'approved')
            self.assertEqual(self.client.delete(f'/knowledge/documents/{doc_id}').status_code, 200)
            self.assertEqual(kb.doc_count, 0)


if __name__ == '__main__':
    unittest.main()
