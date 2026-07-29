"""Regression tests for RAG and human-escalation decision rules."""

import unittest

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentType,
    BaseAgent,
    Request,
)
from api.main import _should_use_knowledge
from core.conversation_rules import is_conversation_memory_query
from core.intent_recognizer import IntentCategory, IntentRecognizer


class KnowledgeDecisionTests(unittest.TestCase):
    def test_skips_greetings_and_small_talk(self):
        for message in ("Hello!", "Thank you", "Okay"):
            with self.subTest(message=message):
                self.assertFalse(_should_use_knowledge(message))

    def test_skips_conversation_memory_questions(self):
        for message in (
            "What did I just ask?",
            "What was my previous question?",
            "Do you remember what I just said?",
            "Summarize our previous conversation",
        ):
            with self.subTest(message=message):
                self.assertFalse(_should_use_knowledge(message))

    def test_uses_rag_for_business_knowledge(self):
        for message in (
            "What is the travel reimbursement process?",
            "How far in advance should I request annual leave?",
            "Who approves system access?",
            "Which approvals are required for a procurement contract?",
        ):
            with self.subTest(message=message):
                self.assertTrue(_should_use_knowledge(message))

    def test_does_not_use_message_length_as_a_signal(self):
        self.assertFalse(_should_use_knowledge("Please introduce yourself and your capabilities"))


class EscalationDecisionTests(unittest.TestCase):
    def test_does_not_escalate_when_merely_describing_human_support(self):
        cases = (
            "I can explain common policies and help you contact the policy owner for complex cases.",
            "Policy exceptions require manual review; I may recommend contacting HR.",
        )
        for content in cases:
            with self.subTest(content=content):
                self.assertFalse(BaseAgent._needs_escalation("Please introduce your capabilities", content))

    def test_escalates_on_explicit_handoff_action(self):
        for message, content in (
            ("I need human support", "I will connect you with human support now."),
            ("I need an above-limit expense exception", "This expense exception requires manual review."),
            ("I need to appeal my attendance result", "The attendance appeal requires human support to verify."),
            ("I need sensitive-data access", "This action requires administrative access."),
        ):
            with self.subTest(message=message, content=content):
                self.assertTrue(BaseAgent._needs_escalation(message, content))


class ConversationRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_question_routes_to_query_without_llm(self):
        recognizer = IntentRecognizer(api_key="not-used-for-this-rule")
        for message in (
            "What was my previous question?",
            "What did I say earlier?",
            "What were we just discussing?",
        ):
            with self.subTest(message=message):
                self.assertTrue(is_conversation_memory_query(message))
                result = await recognizer.recognize(message)
                self.assertEqual(result.intent, IntentCategory.QUERY)
                self.assertEqual(result.confidence, 1.0)
                self.assertEqual(result.latency_ms, 0.0)

    async def test_clear_policy_domains_use_zero_latency_local_routing(self):
        recognizer = IntentRecognizer(api_key="not-used-for-this-rule")
        cases = (
            ("Which receipts are required for travel reimbursement?", IntentCategory.EXPENSE),
            ("How do I request annual leave?", IntentCategory.HR),
            ("How do I apply for system access?", IntentCategory.ACCESS),
            ("转岗后如何申请新系统的数据权限？", IntentCategory.ACCESS),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                result = await recognizer.recognize(message)
                self.assertEqual(result.intent, expected)
                self.assertEqual(result.latency_ms, 0.0)
                self.assertEqual(result.entities, {})

    async def test_entity_extraction_is_disabled_on_the_default_path(self):
        recognizer = IntentRecognizer(api_key="not-used-for-this-rule")

        async def fail_if_called(_message):
            raise AssertionError("entity extraction should be lazy")

        recognizer._extract_entities = fail_if_called
        result = await recognizer.recognize("How do I request annual leave?")
        self.assertEqual(result.intent, IntentCategory.HR)


class CollaborationRoutingTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
        self.orchestrator._pool = {
            AgentType.GENERAL: [object()],
            AgentType.EXPENSE: [object()],
            AgentType.HR: [object()],
            AgentType.ACCESS: [object()],
        }

    def test_routes_cross_domain_question_in_parallel(self):
        request = Request(
            message="I need one day of annual leave during a business trip. How do expense and leave approvals work?",
            user_id="employee-001",
            conv_id="conversation-001",
        )
        targets = self.orchestrator._collaboration_targets(request)
        self.assertEqual(targets, [AgentType.EXPENSE, AgentType.HR])

    def test_routes_access_question_to_access_agent(self):
        request = Request(
            message="How do I request data access to a new system after an internal transfer?",
            user_id="employee-001",
            conv_id="conversation-002",
        )
        targets = self.orchestrator._collaboration_targets(request)
        self.assertEqual(targets, [AgentType.ACCESS])


if __name__ == "__main__":
    unittest.main()
