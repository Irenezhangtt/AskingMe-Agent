"""Integrity guards, independently computed gold labels, and evaluation denominators."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from evaluation.golden.benchmark import DEFAULT_DATA, read_cases, score, summarize, validate
from rag.chunking import RuleChunker
from rag.policy_uploads import UploadedPriceRequest, calculate_uploaded_price
from rag.rule_integrity import condition_kinds


class RuleScopeTests(unittest.TestCase):
    def test_parenthetical_exception_stays_with_its_antecedent(self):
        text = 'Subtract CNY 50 on qualifying orders (except refurbished products (unless Gold status applies)).\n'
        chunks = RuleChunker(5).split(text, parent_id='scope')
        self.assertEqual([chunk.page_content for chunk in chunks], [text])
        self.assertTrue(chunks[0].metadata['oversized'])
        self.assertIn('exception', chunks[0].metadata['condition_kinds'])

    def test_separate_note_does_not_lose_the_rule_it_modifies(self):
        text = 'A CNY 50 discount applies to this order.\n\nExcept clearance products receive no discount.\n'
        chunks = RuleChunker(5).split(text, parent_id='scope')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, text)

    def test_heading_boundary_remains_a_different_scope(self):
        text = '# Appliances\nCNY 50 off (new products only).\n# Furniture\nCNY 20 off (members only).\n'
        chunks = RuleChunker(8).split(text, parent_id='p')
        for chunk in chunks:
            if 'CNY 50' in chunk.page_content:
                self.assertEqual(chunk.metadata['heading_path'], 'Appliances')
                self.assertNotIn('CNY 20', chunk.page_content)
        self.assertEqual(''.join(chunk.page_content for chunk in chunks), text)

    def test_chinese_fullwidth_nested_conditions_are_atomic(self):
        text = '满300减50（仅限会员［家电且非翻新机］，除非有「金卡」）。\n'
        self.assertEqual(len(RuleChunker(3).split(text, parent_id='p')), 1)
        for invalid in ['满300减50（仅限会员］', 'Rule【condition', 'Rule『condition」']:
            with self.assertRaises(ValueError):
                RuleChunker().split(invalid, parent_id='p')

    def test_illustrations_are_labeled_separately_from_conditions(self):
        kinds = condition_kinds('10% off (example only: CNY 500 saves CNY 50, not an extra coupon).')
        self.assertIn('illustration', kinds)
        self.assertIn('negation', kinds)
        self.assertIn('parenthetical', kinds)
        self.assertNotIn('illustration', condition_kinds('10% off (except refurbished items).'))


class GoldenSetTests(unittest.TestCase):
    def test_all_500_labels_and_corpus_references_validate(self):
        report = validate(DEFAULT_DATA)
        self.assertEqual(report['dataset_size'], 500)
        self.assertEqual(report['independent_price_crosschecks'], 390)
        self.assertEqual(report['clarification_cases'], 110)
        self.assertEqual(set(report['families'].values()), {20})

    def test_same_numbers_different_scope_yield_different_answers(self):
        cases = read_cases(DEFAULT_DATA)
        sample = {case['family']: case for case in cases if case['id'].endswith('-02')}
        self.assertEqual(sample['threshold_boundary']['expected']['final_price'], '250.00')
        self.assertEqual(sample['parenthetical_exclusion']['expected']['final_price'], '400.00')
        self.assertEqual(sample['nested_exception']['expected']['final_price'], '360.00')
        self.assertEqual(sample['boolean_grouping']['expected']['final_price'], '370.00')
        self.assertEqual(sample['half_up_rounding']['expected']['final_price'], '0.95')
        self.assertEqual(sample['floor_at_zero']['expected']['final_price'], '0.00')

    def test_scoring_distinguishes_wrong_price_missing_citations_and_provider_error(self):
        case = read_cases(DEFAULT_DATA)[1]
        names = case['expected']['required_source_names']
        result = {'type': 'calculation', 'calculation': {'final_price': case['expected']['final_price'], 'currency': 'CNY'},
                  'answer': 'Final price [1]', 'sources': [{'source_id': i+1, 'source_name': name} for i,name in enumerate(names)], 'excluded': []}
        scored = score(case, result)
        self.assertTrue(scored['pass'])
        self.assertFalse(scored['fully_cited_pass'])
        self.assertAlmostEqual(scored['citation_recall'], 1/3)
        result['calculation']['final_price'] = '0.00'
        self.assertFalse(score(case, result)['pass'])
        rows = [{'expected_type':'calculation','pass':True,'fully_cited_pass':False,'price_exact':True,'latency_ms':10},
                {'expected_type':'calculation','error':'TimeoutError','pass':False,'price_exact':False,'latency_ms':20}]
        self.assertEqual(summarize(rows)['price_exact_rate'], .5)
        self.assertEqual(summarize(rows)['provider_errors'], 1)

    def test_invalid_model_output_is_not_counted_as_correct_abstention(self):
        case = next(case for case in read_cases(DEFAULT_DATA) if case['family']=='missing_membership')
        result = {'type':'clarification','validation_failed':True,'answer':'Could not validate','sources':[],'excluded':[]}
        self.assertFalse(score(case,result)['pass'])

    def test_manifest_hash_matches_published_dataset(self):
        from evaluation.golden.benchmark import sha
        manifest = json.loads(DEFAULT_DATA.with_name('manifest.json').read_text())
        self.assertEqual(manifest['sha256']['cases.jsonl'], sha(DEFAULT_DATA))
        self.assertEqual(manifest['retrieval_evaluable'], 480)


class IntegrityPriceTests(unittest.IsolatedAsyncioTestCase):
    async def test_broken_ocr_brackets_never_reach_price_planner(self):
        case = next(case for case in read_cases(DEFAULT_DATA) if case['family']=='malformed_parentheses')
        request = UploadedPriceRequest.model_validate({'policies':case['policies'],'order':case['order']})
        model = SimpleNamespace(client=SimpleNamespace(messages=SimpleNamespace(create=AsyncMock())))
        result = await calculate_uploaded_price(model, request)
        self.assertEqual(result['type'], 'clarification')
        self.assertIn('unmatched brackets', result['answer'])
        model.client.messages.create.assert_not_awaited()


class RealRerankerContractTests(unittest.TestCase):
    def test_pinned_crossencoder_prediction_keyword(self):
        from unittest.mock import patch
        from rag.models import CrossEncoderReranker
        seen = []
        class FakeCrossEncoder:
            max_length = 512
            tokenizer = SimpleNamespace(encode=lambda *a, **kw: [1, 2])
            def predict(self, pairs, *, activation_fct):
                seen.append((pairs, activation_fct))
                return [.9, .4]
        reranker = CrossEncoderReranker()
        reranker._model = FakeCrossEncoder()
        # Patch import-time heavy dependencies; the signature matches pinned 3.4.1.
        with patch.dict('sys.modules', {'sentence_transformers':SimpleNamespace(CrossEncoder=FakeCrossEncoder),
                                       'torch':SimpleNamespace(nn=SimpleNamespace(Sigmoid=lambda:'sigmoid'))}):
            ranked = reranker._rank('q', [{'chunk_id':'a','content':'policy'}, {'chunk_id':'b','content':'other'}])
        self.assertEqual(ranked[0]['chunk_id'], 'a')
        self.assertEqual(seen[0][1], 'sigmoid')


class ConditionExplanationTests(unittest.TestCase):
    def test_scope_checks_are_rendered_with_validated_citations(self):
        from rag.llm import render_answer_plan
        plan={'type':'calculation','expression':'400','currency':'CNY','source_ids':[1],
              'rule_checks':[{'condition':'Refurbished-product exception','outcome':'excluded',
                              'reason':'The item is refurbished, so the CNY 50 benefit is excluded.','source_ids':[1]}]}
        answer=render_answer_plan(plan,[{'content':'CNY 50 off (except refurbished products).'}])
        self.assertIn('Excluded: Refurbished-product exception',answer)
        self.assertIn('so the CNY 50 benefit is excluded. [1]',answer)
        plan['rule_checks'][0]['source_ids']=[2]
        with self.assertRaises(ValueError):render_answer_plan(plan,[{'content':'rule'}])


class ProviderCreditTests(unittest.TestCase):
    def test_credit_exhaustion_is_classified_without_exporting_provider_messages(self):
        from evaluation.golden.benchmark import provider_blocker
        error=RuntimeError('sensitive endpoint data')
        error.body={'error':{'message':'Your credit balance is too low to access the Anthropic API.'}}
        self.assertEqual(provider_blocker(error),'credit_balance_exhausted')
        self.assertIsNone(provider_blocker(RuntimeError('a different failure')))

    def test_not_run_cases_are_not_counted_as_successful_responses(self):
        summary=summarize([{'id':'x','expected_type':'calculation','not_run':True,'pass':False,'latency_ms':0}])
        self.assertEqual(summary['completed_responses'],0)
        self.assertEqual(summary['not_run'],1)
        self.assertEqual(summary['provider_errors'],0)
        self.assertEqual(summary['pass_rate'],0)
