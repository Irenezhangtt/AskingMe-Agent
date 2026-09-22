"""Final-price tool correctness and agent/tool contract tests, without paid API calls."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from rag.calculator import CalculationError, calculate_final_price
from rag.llm import PricingLLM, render_answer_plan


class FinalPriceTests(unittest.TestCase):
    def test_supported_pricing_scenarios(self):
        cases = {
            '(400 - 50 - 20) * 0.9': '297.00',
            '400 - 50 - 20': '330.00',
            '(300 - 50 - 20) * 0.9': '207.00',
            '(299.99 - 20) * 0.9': '251.99',
            '(600 - 50 - 20) * 0.9': '477.00',
            'max(0, 10 - 20) * 0.9': '0.00',
            'min(20, 50) + 0.1 + 0.2': '20.30',
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculate_final_price(expression)['final_price'], expected)

    def test_half_up_and_no_intermediate_cent_rounding(self):
        self.assertEqual(calculate_final_price('2.675')['final_price'], '2.68')
        self.assertEqual(calculate_final_price('1.005 + 1.005')['final_price'], '2.01')
        self.assertEqual(calculate_final_price('0.1 + 0.2')['unrounded'], '0.3')
        self.assertEqual(calculate_final_price('1 / 3 * 3')['final_price'], '1.00')

    def test_rejects_unsupported_or_unsafe_expressions(self):
        cases = ["__import__('os').system('echo unsafe')", 'price - 20', '10 ** 1000',
                 '1 / 0', '-1', 'True', '1e5', 'min(1)', 'max(x=1, y=2)',
                 '(1).__class__', '[1][0]', '1000000000000 * 1000000000000', '']
        for expression in cases:
            with self.subTest(expression=expression), self.assertRaises(CalculationError):
                calculate_final_price(expression)

    def test_audit_trail_contains_computed_intermediate_values(self):
        result = calculate_final_price('(400 - 50 - 20) * 0.9')
        self.assertEqual([s['result'] for s in result['steps']], ['350', '330', '297.0'])
        self.assertEqual(result['rounding'], 'ROUND_HALF_UP')

    def test_evidence_references_are_required(self):
        plan = {'type': 'calculation', 'expression': '400 - 50', 'currency': 'CNY', 'source_ids': [2]}
        with self.assertRaises(ValueError):
            render_answer_plan(plan, [{'content': 'rules'}])
        plan['source_ids'] = [True]
        with self.assertRaises(ValueError):
            render_answer_plan(plan, [{'content': 'rules'}])


class FinalPriceAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_computes_and_streams_the_validated_amount(self):
        plan = {'type': 'calculation', 'expression': '(400 - 50 - 20) * 0.9', 'currency': 'CNY', 'source_ids': [1]}
        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type='text', text=json.dumps(plan))]))))
        agent = PricingLLM(client, 'test-model')
        callback = AsyncMock()
        answer = await agent.answer('Calculate my final price', [{'content': 'test rules'}], token_callback=callback)
        self.assertIn('Final price: CNY 297.00', answer)
        self.assertEqual(''.join(call.args[0] for call in callback.call_args_list), answer)
        self.assertIn('Decimal calculation tool', client.messages.create.call_args.kwargs['system'])

    async def test_invalid_plan_does_not_emit_a_price(self):
        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type='text', text='{"type":"calculation","expression":"1/0","currency":"CNY","source_ids":[1]}')]))))
        answer = await PricingLLM(client, 'test').answer('Calculate', [{'content': 'rules'}])
        self.assertIn('No verified final price', answer)

    async def test_missing_input_clarification_does_not_invent_a_price(self):
        plan = {'type': 'clarification', 'answer': 'Is your coupon valid and already claimed?'}
        client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=SimpleNamespace(
            content=[SimpleNamespace(type='text', text=json.dumps(plan))]))))
        answer = await PricingLLM(client, 'test').answer('Calculate', [{'content': 'rules'}])
        self.assertEqual(answer, plan['answer'])
