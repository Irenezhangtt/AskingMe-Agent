"""
Multi-agent routing and orchestration.

The three-layer routing strategy combines intent routing, performance-based
selection among same-type agents, and fallback to GeneralAgent. Compound
questions can run across several agents in parallel before the orchestrator
merges their responses. Low-confidence or sensitive cases can be escalated.
"""
import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # General policy Q&A
    EXPENSE   = "expense"    # Expense and travel policies
    HR        = "hr"         # Human resources policies
    ACCESS    = "access"     # Access and internal systems
    ESCALATION = "escalation" # Human escalation placeholder


@dataclass
class AgentStats:
    """Runtime agent statistics used by monitoring and routing decisions."""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """Score agents higher for strong success rates and low latency."""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # Whether escalation is required


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # Formatted context from MemoryManager
    history:     Optional[List[Dict[str, str]]] = None  # Conversation history for intent recognition
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    token_callback: Optional[Callable[[str], Awaitable[None]]] = None
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    timings:     Dict[str, float] = field(default_factory=dict)


# ── Base agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """Base class that encapsulates LLM calls and runtime statistics."""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(req.message, content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} processing failed: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="Sorry, something went wrong while processing your request. Please try again shortly.",
                success=False,
                latency_ms=ms,
            )

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[Background]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "Understood. I have reviewed the background information."})
        messages.append({"role": "user", "content": _clean(req.message)})

        request_args = {
            "model": self._model,
            "max_tokens": max(128, int(os.getenv("ANSWER_MAX_TOKENS", "650"))),
            "system": self._build_system_prompt(req),
            "messages": messages,
        }
        if req.token_callback is not None:
            chunks: List[str] = []
            async with self._client.messages.stream(**request_args) as stream:
                async for text in stream.text_stream:
                    chunks.append(text)
                    await req.token_callback(text)
            return "".join(chunks)

        resp = await self._client.messages.create(**request_args)
        return extract_text_content(resp.content)

    def _build_system_prompt(self, req: Request) -> str:
        """Append dynamically loaded skills to the system prompt."""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[Dynamic Skills]\n{skill_prompt}"

    @staticmethod
    def _needs_escalation(user_message: str, content: str) -> bool:
        """Determine whether the current request and response require a human handoff.

        Merely describing situations that require a human does not escalate the
        current conversation. Escalate only for an explicit handoff or when both
        the active request and response confirm manual handling is required.
        """
        message = " ".join((user_message or "").lower().split())
        normalized = " ".join((content or "").lower().split())

        # Explicit current handoff actions do not depend on business keywords.
        immediate_handoff_phrases = [
            "connect you with a person",
            "connect you with human support",
            "escalate this to a person",
            "escalated to human support",
            "escalate to a human",
            "escalate to a specialist",
        ]
        if any(phrase in normalized for phrase in immediate_handoff_phrases):
            return True

        # Current tasks that require human execution or approval must match both
        # the user request and response confirmation to avoid false positives.
        manual_case_keywords = [
            "above-limit expense", "missing receipt", "cash advance", "expense exception", "travel exception",
            "leave dispute", "attendance appeal", "attendance result", "employment contract", "termination certificate", "pay dispute",
            "provision access", "elevate access", "data export", "sensitive data", "sensitive-data", "access anomaly",
            "policy conflict", "compliance", "legal", "audit",
        ]
        manual_confirmation_phrases = [
            "requires human support",
            "recommend escalation",
            "contact human support",
            "contact the policy owner",
            "contact hr",
            "contact an administrator",
            "requires manual review",
            "requires manual verification",
            "requires manual handling",
            "requires human intervention",
            "cannot process this directly",
            "cannot perform this operation",
            "outside my authority",
            "requires administrative access",
        ]
        has_manual_case = any(keyword in message for keyword in manual_case_keywords)
        has_manual_confirmation = any(phrase in normalized for phrase in manual_confirmation_phrases)
        return has_manual_case and has_manual_confirmation


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    system_prompt = (
        "You are AskingMe Agent, an internal company policy assistant for employees. "
        "Base answers on current knowledge-base provisions and clearly state scope, process, and required documents. "
        "If information is missing, versions conflict, or actual approval is required, say so and recommend the responsible policy owner."
    )


class ExpenseAgent(BaseAgent):
    agent_type    = AgentType.EXPENSE
    system_prompt = (
        "You are an enterprise expense and travel policy specialist. Focus on eligible expenses, travel limits, "
        "receipt requirements, approval chains, and exceptions. Cite knowledge-base policy and never invent limits or promise approval."
    )


class HRAgent(BaseAgent):
    agent_type    = AgentType.HR
    system_prompt = (
        "You are an enterprise HR policy specialist. Focus on leave, attendance, benefits, onboarding, offboarding, and employment policy. "
        "Distinguish policy explanation from approval, do not predict individual outcomes, and refer sensitive matters to HR."
    )


class AccessAgent(BaseAgent):
    agent_type    = AgentType.ACCESS
    system_prompt = (
        "You are an internal systems and data-access policy specialist. Focus on account provisioning, role permissions, "
        "access approval, least privilege, and sensitive-data rules. Never claim to have performed an administrator action."
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """Coordinate agents through intent, performance, and fallback routing."""

    # Static intent-to-agent-type mapping (routing table)
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.EXPENSE:    AgentType.EXPENSE,
        IntentCategory.HR:         AgentType.HR,
        IntentCategory.ACCESS:     AgentType.ACCESS,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        # All other intents map to GENERAL by default
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-sonnet-4-6",
        router_model: Optional[str] = None,
        skill_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(
            api_key=api_key,
            base_url=base_url,
            model=router_model or model,
        )
        self._skill_manager = skill_manager

        # Agent pool: each type may have multiple instances for horizontal scaling
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [GeneralAgent(client, model, skill_manager)],
            AgentType.EXPENSE: [ExpenseAgent(client, model, skill_manager)],
            AgentType.HR:      [HRAgent(client, model, skill_manager)],
            AgentType.ACCESS:  [AccessAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """Update SkillManager references for runtime reloads or test substitution."""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """Run intent recognition, routing, execution, and escalation checks."""
        t0 = time.monotonic()
        intent_ms = 0.0

        # 1. Recognize intent unless the caller already supplied it
        if req.intent is None:
            intent_t0 = time.monotonic()
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            intent_ms = (time.monotonic() - intent_t0) * 1000
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency

        # Collaborate in parallel for compound questions spanning multiple policy domains.
        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            result = await self.run_parallel(req, collaboration)
            result.timings["intent_ms"] = round(intent_ms, 1)
            result.latency_ms = (time.monotonic() - t0) * 1000
            return result

        # 2. Route to the selected agent type
        agent_type = self._route(req.intent, req.urgency)

        # 3. Execute with fallback
        answer_t0 = time.monotonic()
        response = await self._execute(req, agent_type)
        answer_ms = (time.monotonic() - answer_t0) * 1000

        # 4. Check whether human escalation is required
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"Request {req.request_id} triggered escalation: urgency={req.urgency}")
            # Production: create an internal ticket and notify the policy owner here.

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            timings={
                "intent_ms": round(intent_ms, 1),
                "answer_ms": round(answer_ms, 1),
            },
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """Dispatch to several agents in parallel and merge their responses."""
        t0 = time.monotonic()
        # Do not interleave tokens from several specialists. Stream the merged
        # response after all parallel calls finish instead.
        parallel_req = replace(req, token_callback=None)
        tasks = [self._execute(parallel_req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge all successful responses
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "Sorry, every agent failed to process the request."
        if req.token_callback is not None:
            chunk_size = max(16, int(os.getenv("STREAM_CHUNK_CHARS", "48")))
            for start in range(0, len(combined), chunk_size):
                await req.token_callback(combined[start:start + chunk_size])
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            timings={"answer_ms": round((time.monotonic() - t0) * 1000, 1)},
        )

    # ── Routing logic ─────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """Apply intent mapping, critical-urgency override, and GENERAL fallback."""
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # Use the requested type when available; otherwise fall back
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """Detect compound questions that require several agents in parallel."""
        msg = req.message.lower()
        targets: List[AgentType] = []

        expense_kws = [
            "expense", "reimbursement", "travel", "receipt", "hotel", "transportation",
            "meal allowance", "procurement", "contract", "报销", "差旅", "出差", "发票", "采购", "合同",
        ]
        hr_kws = [
            "leave", "annual leave", "sick leave", "attendance", "onboarding",
            "offboarding", "benefits", "overtime", "请假", "年假", "病假", "考勤", "入职", "离职", "加班",
        ]
        access_kws = [
            "access", "account", "permission", "provision", "data", "role", "system",
            "权限", "账号", "账户", "数据访问", "系统访问",
        ]

        if req.intent == IntentCategory.EXPENSE or any(kw in msg for kw in expense_kws):
            targets.append(AgentType.EXPENSE)
        if req.intent == IntentCategory.HR or any(kw in msg for kw in hr_kws):
            targets.append(AgentType.HR)
        if req.intent == IntentCategory.ACCESS or any(kw in msg for kw in access_kws):
            targets.append(AgentType.ACCESS)

        # Deduplicate while preserving order and return only types with active instances.
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """Select the same-type agent with the highest live routing score."""
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """Execute an agent and fall back to GeneralAgent on failure."""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="The service is temporarily unavailable. Please try again shortly.",
                success=False,
            )

        response = await agent.handle(req)

        # Fall back to GeneralAgent when a specialized agent fails
        if not response.success and agent_type != AgentType.GENERAL:
            logger.warning(f"{agent_type.value} failed; falling back to GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                response = await fallback.handle(req)

        return response

    # ── Statistics consumed by the monitor ────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """Apply monitor feedback as dynamic routing penalties keyed like access_0."""
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
