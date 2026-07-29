"""End-to-end intent recognition with weighted LLM, embedding, and pattern strategies."""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.conversation_rules import is_conversation_memory_query
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    QUERY      = "query"       # Information query
    COMPLAINT  = "complaint"   # Complaint or dissatisfaction
    REQUEST    = "request"     # Action request
    GREETING   = "greeting"    # Greeting
    ESCALATION = "escalation"  # Human escalation request
    EXPENSE    = "expense"     # Expense and travel policies
    HR         = "hr"          # HR, leave, and attendance policies
    ACCESS     = "access"      # System account and access policies
    FEEDBACK   = "feedback"    # Positive feedback
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # Entities extracted from the message
    reasoning:  str
    latency_ms: float


# ── Few-shot templates used by both LLM examples and embedding matching ───────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["What is the remote-work policy?", "When does the policy take effect?", "Which employees are covered?"],
    IntentCategory.COMPLAINT:  ["My approval has been pending too long", "This process is unreasonable", "The policy statements conflict"],
    IntentCategory.REQUEST:    ["Help me confirm the approval workflow", "I need to request system access", "Tell me which documents are required"],
    IntentCategory.GREETING:   ["Hello", "Hi, is anyone there?", "Good morning"],
    IntentCategory.ESCALATION: ["Connect me with the policy owner", "I need to contact HR", "A manager must confirm this"],
    IntentCategory.EXPENSE:    ["How do I claim travel expenses?", "What is the hotel allowance?", "Which receipts are required?"],
    IntentCategory.HR:         ["How do I request annual leave?", "What evidence is required for sick leave?", "How do I correct attendance?"],
    IntentCategory.ACCESS:     ["How do I request system access?", "I cannot access the internal system", "Who approves data access?"],
    IntentCategory.FEEDBACK:   ["That was helpful", "The policy explanation was clear", "Thank you"],
}

# Urgency keywords
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["emergency", "urgent", "asap", "immediately"],
    UrgencyLevel.HIGH:     ["today", "right away", "as soon as possible", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["this week", "soon", "quickly"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity in pure Python without NumPy."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """Recognize intents with remote LLM calls and lazily cached template embeddings."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # Third-party compatible APIs often lack embeddings, so disable that strategy.
        # The official Anthropic SDK currently has no embeddings resource; use stable
        # local character n-gram vectors as a lightweight fallback.
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── Public interface ───────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """Recognize intent using optional role/content conversation history."""
        # Conversation-recall questions are deterministic queries. Route them to
        # GeneralAgent to avoid policy keywords in history causing misrouting.
        if is_conversation_memory_query(message):
            return IntentResult(
                intent=IntentCategory.QUERY,
                confidence=1.0,
                urgency=UrgencyLevel.LOW,
                entities={},
                reasoning="The user is recalling the current conversation",
                latency_ms=0.0,
            )

        # Most employee questions contain an unambiguous policy-domain signal.
        # Resolve those locally so the common path does not spend two remote LLM
        # calls on intent classification and entity extraction.
        fast_intent = self._fast_intent(message)
        if fast_intent is not None:
            intent, reasoning = fast_intent
            return IntentResult(
                intent=intent,
                confidence=0.98,
                urgency=self._urgency(message, intent),
                entities={},
                reasoning=reasoning,
                latency_ms=0.0,
            )

        key = self._cache_key(message)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # Run LLM and embedding strategies in parallel when embeddings are available
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent = self._vote(llm, emb, pat)
        entities = {}
        if os.getenv("INTENT_EXTRACT_ENTITIES", "false").lower() == "true":
            entities = await self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=llm["confidence"],
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
        )

        # LRU cache
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    @staticmethod
    def _fast_intent(message: str) -> Optional[tuple[IntentCategory, str]]:
        """Return a deterministic intent for clear, high-confidence messages."""
        msg = " ".join((message or "").lower().split())
        if not msg:
            return IntentCategory.OTHER, "Empty message"

        exact_greetings = {
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "你好", "您好", "早上好", "下午好",
        }
        exact_feedback = {
            "thanks", "thank you", "that was helpful", "谢谢", "感谢",
        }
        if msg.strip(" ,.!?，。！？") in exact_greetings:
            return IntentCategory.GREETING, "Matched a greeting locally"
        if msg.strip(" ,.!?，。！？") in exact_feedback:
            return IntentCategory.FEEDBACK, "Matched feedback locally"

        escalation_terms = (
            "human support", "contact hr", "policy owner", "escalate",
            "人工", "联系hr", "联系 hr", "制度负责人", "升级处理",
        )
        if any(term in msg for term in escalation_terms):
            return IntentCategory.ESCALATION, "Matched an explicit handoff request locally"

        domain_terms = {
            IntentCategory.EXPENSE: (
                "expense", "reimbursement", "business trip", "travel", "receipt",
                "hotel", "transportation", "meal allowance", "procurement", "contract",
                "报销", "差旅", "出差", "发票", "住宿", "交通", "餐补", "采购", "合同",
            ),
            IntentCategory.HR: (
                "annual leave", "sick leave", "personal leave", "attendance",
                "timecard", "overtime", "benefits", "onboarding", "offboarding",
                "请假", "年假", "病假", "事假", "考勤", "打卡", "加班", "福利", "入职", "离职",
            ),
            IntentCategory.ACCESS: (
                "system access", "data access", "permission", "account", "provision",
                "least privilege", "系统权限", "数据权限", "访问权限", "账号", "账户", "开通权限",
            ),
        }
        scores = {
            intent: sum(1 for term in terms if term in msg)
            for intent, terms in domain_terms.items()
        }
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] > 0:
            return best_intent, f"Matched the {best_intent.value} policy domain locally"
        return None

    def learn(self, message: str, correct: IntentCategory) -> None:
        """Learn a corrected example and invalidate its embedding cache."""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # Recompute on the next request
            logger.info(f"Learned new sample -> {correct.value}: {message[:40]}")

    # ── Three-way recognition strategy ────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """Strategy 1: few-shot LLM semantic understanding with context."""
        message = self._clean_text(message)
        # Build few-shot examples
        examples = "\n".join(
            f'  Message: "{t}" -> intent: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # Use one example per class to control prompt length
        )
        # Context from the three most recent turns
        ctx = ""
        if history:
            ctx = "\nRecent conversation:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""You are an intent-analysis expert for an internal company policy assistant. Determine the employee's intent and return JSON.

Examples:
{examples}

{ctx}
User message: "{message}"

Return format (JSON only):
{{"intent": "<intent value>", "confidence": <0-1>, "reasoning": "<one-sentence explanation>"}}

Allowed intents: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM recognition failed: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM failed", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """Strategy 2: embedding-vector similarity matching."""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding recognition failed: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """Strategy 3: synchronous keyword-pattern fallback."""
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["policy owner", "contact hr", "manager confirmation", "human support", "escalate"],
            IntentCategory.COMPLAINT:  ["unreasonable", "conflict", "nobody has handled", "appeal"],
            IntentCategory.QUERY:      ["?", "how", "what", "which", "status"],
            IntentCategory.REQUEST:    ["help me", "i need", "please", "help"],
            IntentCategory.GREETING:   ["hello", "hi", "good morning", "good evening"],
            IntentCategory.EXPENSE:    ["expense", "reimbursement", "travel", "receipt", "hotel allowance"],
            IntentCategory.HR:         ["leave", "annual leave", "sick leave", "attendance", "onboarding", "offboarding", "benefits"],
            IntentCategory.ACCESS:     ["access", "account", "permission", "provision", "data access", "system"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── Voting and merge ───────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> IntentCategory:
        """Merge strategies by weighted vote and redistribute unavailable embedding weight."""
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"]
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"]
            return IntentCategory.OTHER

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentCategory.OTHER

    # ── Entity extraction ──────────────────────────────────────────────────────

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """Extract structured entities from a message with the LLM."""
        message = self._clean_text(message)
        prompt = f"""Extract entities from an employee's company-policy question. Return JSON with list values; use an empty list when absent.
Message: "{message}"
Format: {{"policy_domain":[],"department":[],"date":[],"amount":[],"system":[],"approval_role":[]}}"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {
                "policy_domain": [],
                "department": [],
                "date": [],
                "amount": [],
                "system": [],
                "approval_role": [],
            }

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """Lazily load embeddings for all templates on first use."""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """Generate a remote embedding when available or use local n-gram hashing."""
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"Remote embedding failed; using the local vector fallback: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """Build a stable character n-gram hash vector for local approximate matching."""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str) -> str:
        return self._clean_text(message)[:200]

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Remove Unicode surrogate characters before HTTP prompt encoding."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
