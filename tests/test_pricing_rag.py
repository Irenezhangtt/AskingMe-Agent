"""Offline regression tests; actual LangGraph, fake external transports/models."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from rag.chunking import RuleChunker
from rag.elasticsearch_store import ElasticsearchKnowledgeBase, filter_clauses, reciprocal_rank_fusion
from rag.graph import PricingRAG, select_context
from evaluation.rag_benchmark import AssertionJudge, assertion_metrics, load_dataset, retrieval_metrics


def document(key='a', content='CNY 50 off a subtotal of CNY 300 or more, once per order.', score=.9):
    return {'chunk_id': key, 'document_id': key, 'content': content, 'score': score, 'source_name': 'example.md'}


class ChunkingTests(unittest.TestCase):
    def test_preserves_nested_formula_and_hierarchy(self):
        formula = 'Price = A - (B if (C and D) else 0) + E\n'
        text = '# Discount rules\n## Member discount\nExplanation.\n' + formula + 'Other rules. ' * 30
        docs = RuleChunker(20).split(text, parent_id='parent', title='Rules')
        self.assertTrue(any(formula in d.page_content for d in docs))
        self.assertTrue(any(d.metadata['heading_path'] == 'Discount rules > Member discount' for d in docs))
        self.assertTrue(all(d.metadata['parent_id'] == 'parent' for d in docs))
        self.assertEqual(''.join(d.page_content for d in docs), text)

    def test_english_sentences_preserve_decimal_boundaries(self):
        text = 'The subtotal is CNY 299.99. Members receive a discount.'
        cuts = RuleChunker._boundaries(text)
        decimal = text.index('299.99') + 4
        sentence = text.index('. Members') + 1
        self.assertNotEqual(cuts.get(decimal), 1)
        self.assertEqual(cuts[sentence], 1)
        docs = RuleChunker().split('If the subtotal is at least CNY 300, subtract CNY 50.', parent_id='p')
        self.assertEqual(docs[0].metadata['formula_type'], 'conditional')

    def test_oversize_formula_is_flagged_not_truncated(self):
        text = 'Price = (' + 'A + ' * 30 + 'B)\n'
        docs = RuleChunker(10).split(text, parent_id='p')
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0].metadata['oversized'])
        self.assertEqual(docs[0].page_content, text)

    def test_rejects_malformed_rules_even_when_short(self):
        for text in ['Price = (A - B', 'Price = A - B)', 'Rule（unclosed', '```python\nA = B\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                RuleChunker().split(text, parent_id='p')

    def test_fenced_code_is_atomic_and_headings_inside_are_not_sections(self):
        text = '# Rules\n```python\n# not a heading\nP = (A - B)\n```\nText.\n'
        docs = RuleChunker(12).split(text, parent_id='p')
        code = next(d for d in docs if '```' in d.page_content)
        self.assertEqual(code.page_content.count('```'), 2)
        self.assertTrue(all(d.metadata['heading_path'] == 'Rules' for d in docs))


class RetrievalTests(unittest.TestCase):
    def test_rrf_uses_rank_and_deduplicates(self):
        a, b, c = document('a', score=10000), document('b', score=.01), document('c')
        result = reciprocal_rank_fusion([a, b], [b, c])
        self.assertEqual(result[0]['chunk_id'], 'b')
        self.assertAlmostEqual(result[0]['rrf_score'], 1/62 + 1/61)
        self.assertEqual(len(reciprocal_rank_fusion([a, a], [a])), 1)

    def test_approved_filter_cannot_be_overridden(self):
        with self.assertRaises(ValueError):
            filter_clauses({'status': 'draft'})
        filters = filter_clauses({'category': 'Appliances', 'as_of': '2026-09-22'})
        self.assertIn({'term': {'status': 'approved'}}, filters)
        self.assertIn({'term': {'category': 'Appliances'}}, filters)
        self.assertEqual(len(filters), 4)

    def test_same_filter_on_both_retrieval_branches(self):
        kb = ElasticsearchKnowledgeBase.__new__(ElasticsearchKnowledgeBase)
        kb.client, kb.embeddings, kb.index = MagicMock(), MagicMock(), 'test'
        kb.client.search.return_value = {'hits': {'hits': []}}
        kb.embeddings.embed_query.return_value = [.1, .2]
        kb.retrieve('CNY 50 off orders of CNY 300 or more', filters={'event': 'Demo Promotion'})
        calls = [c.kwargs for c in kb.client.search.call_args_list]
        lex = next(c for c in calls if 'query' in c)
        vec = next(c for c in calls if 'knn' in c)
        self.assertEqual(lex['query']['bool']['filter'], vec['knn']['filter']['bool']['filter'])
        self.assertEqual(vec['knn']['num_candidates'], 500)

    def test_dynamic_context_filters_weak_hits_and_never_truncates(self):
        rows = [document('a', score=.9), document('b', score=.8), document('c', score=.4)]
        self.assertEqual([r['chunk_id'] for r in select_context(rows)], ['a', 'b'])
        long = document('long', content='Rule. ' * 1000)
        self.assertEqual(select_context([long], token_budget=100), [])
        self.assertEqual(select_context([document(score=.1)]), [])


class GraphTests(unittest.IsolatedAsyncioTestCase):
    def make_rag(self, rows, scores):
        store = MagicMock()
        store.retrieve.return_value = rows
        reranker = MagicMock()
        reranker.rank = AsyncMock(side_effect=scores)
        llm = MagicMock()
        llm.rewrite = AsyncMock(return_value='Eligibility for CNY 50 off orders of CNY 300 or more')
        llm.answer = AsyncMock(return_value='The discount applies once per order. [1]')
        return PricingRAG(store, reranker, llm)

    async def test_strong_evidence_skips_rewrite(self):
        rows = [document()]
        rag = self.make_rag(rows, [rows])
        result = await rag.run('How does CNY 50 off orders of CNY 300 or more work?')
        rag.llm.rewrite.assert_not_awaited()
        self.assertTrue(result['sufficient'])
        self.assertEqual([x['stage'] for x in result['trace']], ['query_analysis', 'hybrid_retrieval', 'rerank', 'response_generation'])

    async def test_weak_evidence_rewrites_once_then_recovers(self):
        rows = [document()]
        rag = self.make_rag(rows, [[document(score=.1)], rows])
        result = await rag.run('How does CNY 50 off orders of CNY 300 or more work?', filters={'category': 'Appliances'})
        self.assertEqual(result['attempts'], 1)
        self.assertTrue(result['sufficient'])
        self.assertEqual(rag.store.retrieve.call_count, 2)
        self.assertTrue(all(c.args[2] == {'category': 'Appliances'} for c in rag.store.retrieve.call_args_list))

    async def test_empty_evidence_stops_and_streams_abstention(self):
        rag = self.make_rag([], [[], []])
        callback = AsyncMock()
        result = await rag.run('How does CNY 50 off orders of CNY 300 or more work?', token_callback=callback)
        self.assertFalse(result['sufficient'])
        rag.llm.answer.assert_not_awaited()
        self.assertEqual(rag.store.retrieve.call_count, 2)
        callback.assert_awaited_once_with(result['answer'])

    async def test_search_only_never_generates(self):
        rows = [document()]
        rag = self.make_rag(rows, [rows])
        result = await rag.run('Discount', generate=False)
        self.assertNotIn('answer', result)
        rag.llm.answer.assert_not_awaited()

    async def test_rewrite_cannot_change_numbers(self):
        rows = [document()]
        rag = self.make_rag(rows, [[], rows])
        rag.llm.rewrite.return_value = 'CNY 100 off orders of CNY 500 or more'
        await rag.run('CNY 50 off orders of CNY 300 or more')
        self.assertEqual(rag.store.retrieve.call_args_list[1].args[0], 'CNY 50 off orders of CNY 300 or more')

    async def test_ablation_has_no_reranking_or_rewriting(self):
        rag = self.make_rag([document(score=.01)], [])
        result = await rag.run('Discount', mode='dense', use_rerank=False, use_rewrite=False, generate=False)
        self.assertEqual(len(result['documents']), 1)
        rag.reranker.rank.assert_not_awaited()
        rag.llm.rewrite.assert_not_awaited()

    async def test_state_is_isolated_between_requests(self):
        rows = [document()]
        rag = self.make_rag(rows, [rows, rows])
        await rag.run('First question')
        result = await rag.run('Second question')
        self.assertEqual(result['question'], 'Second question')
        self.assertEqual(len(result['trace']), 4)


class EvaluationTests(unittest.TestCase):
    def test_recall_does_not_count_duplicate_chunks(self):
        metrics = retrieval_metrics(['a', 'a', 'irrelevant', 'b'], ['a', 'b', 'c'])
        self.assertAlmostEqual(metrics['recall@5'], 2/3)
        self.assertEqual(metrics['mrr@5'], 1)

    def test_assertion_denominators_are_distinct(self):
        metrics = assertion_metrics([True, True, False], [True, True, False, False, False], [True] * 5)
        self.assertAlmostEqual(metrics['faithfulness'], 2/3)
        self.assertEqual(metrics['answer_claim_recall'], .4)
        self.assertEqual(metrics['retrieved_claim_recall'], 1)
        self.assertIsNone(assertion_metrics([], [], [])['faithfulness'])
        with self.assertRaises(ValueError):
            assertion_metrics(['false'], [], [])

    def test_sample_dataset_is_small_and_explicitly_labeled(self):
        cases = load_dataset('examples/evaluation/pricing.jsonl')
        self.assertEqual(len(cases), 8)
        from pathlib import Path
        for case in cases:
            for source in case['relevant_documents']:
                self.assertTrue((Path('examples/pricing') / source).is_file(), source)


class JudgeFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_is_missing_data_not_fabricated_score(self):
        judge = AssertionJudge(MagicMock(), 'test')
        judge._json = AsyncMock(side_effect=RuntimeError('offline'))
        result = await judge.evaluate('q', 'a', 'context', ['fact'])
        self.assertTrue(result['judge_failed'])
        self.assertIsNone(result['faithfulness'])


if __name__ == '__main__':
    unittest.main()
