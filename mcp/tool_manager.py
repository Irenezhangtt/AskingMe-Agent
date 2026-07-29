"""MCP tools with query rewriting, reranking, caching, fallback, and circuit breaking."""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # Normal operation
    OPEN      = "open"       # Reject requests while tripped
    HALF_OPEN = "half_open"  # Probe for recovery


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # Whether results were reranked


@dataclass
class ToolStats:
    """Tool runtime statistics consumed by the monitor."""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── Circuit breaker ───────────────────────────────────────────────────────────

class CircuitBreaker:
    """Three-state circuit breaker with timed half-open recovery probes."""

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN allows one recovery probe

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"Circuit breaker opened after {self.fail_count} consecutive failures")


# ── Tool definition ───────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 disables caching
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # Whether result reranking is supported
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any

    # Runtime state excluded from construction
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP tool manager ──────────────────────────────────────────────────────────

class MCPToolManager:
    """Manage MCP tools and the optimized retrieval pipeline."""

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at, reranked)

    # ── Registration and removal ───────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── Core invocation ────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,          # Rerank and return top K when greater than zero
    ) -> ToolResult:
        """Invoke a tool through cache, breaker, validation, timeout, reranking, and write-back."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"Tool does not exist: {name}")

        cache_rerank_top_k = rerank_top_k if rerank_top_k > 0 and tool.supports_rerank else 0

        # Cache hit
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, cache_rerank_top_k)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached_data,
                    tool_name=name,
                    cached=True,
                    reranked=cached_reranked,
                )

        # Circuit-breaker check
        if not tool.breaker.allow():
            error = f"Tool circuit is open: {name}. Please try again later."
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # Validate arguments against JSON Schema required fields and property types
            self._validate_params(tool, params)

            data = await asyncio.wait_for(tool.handler(params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            # Rerank list results returned by retrieval tools
            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k), True

            # Cache the final reranked result rather than the raw retrieval output.
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl, cache_rerank_top_k, reranked)

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"Tool timed out: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "Execution timed out")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"Tool error: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """Return a meaningful fallback result when a tool is unavailable."""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"Tool fallback failed: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback failed: {ex}")

    # ── Query rewriting to improve recall ──────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """Rewrite one query into several perspectives to improve document recall."""
        prompt = f"""Rewrite the following user query into {n} search subqueries for knowledge-base retrieval.
Each subquery must take a different angle and cover a distinct aspect of the original question.
Original query: "{query}"
Return a JSON array, for example: ["subquery 1", "subquery 2", "subquery 3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            # Preserve and deduplicate the original query
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"Query rewriting failed; using the original query: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Run rewriting, parallel retrieval, deduplication, reranking, and top-K selection."""
        # 1. Rewrite the query into multiple perspectives
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"Query rewrite: {query!r} -> {sub_queries}")

        # 2. Retrieve all subqueries in parallel
        recall_k = max(top_k, 5)
        tasks = [
            self.call(tool_name, {"query": q, "top_k": recall_k}, context, use_cache=True)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Merge and deduplicate by content hash
        seen, merged = set(), []
        for r in results:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    key = hashlib.md5(str(item).encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)

        if not merged:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="No subquery returned results")

        # 4. Use the LLM to rerank merged results and return the top K
        rerank_mode = os.getenv("RAG_RERANK_MODE", "adaptive").lower()
        merged.sort(
            key=lambda item: float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
            reverse=True,
        )
        if rerank_mode == "never" or (
            rerank_mode == "adaptive" and not self._results_need_rerank(merged, top_k)
        ):
            return ToolResult(
                success=True,
                data=merged[:top_k],
                tool_name=tool_name,
                reranked=False,
            )

        reranked = await self._rerank(query, merged, top_k)
        return ToolResult(success=True, data=reranked, tool_name=tool_name, reranked=True)

    @staticmethod
    def _results_need_rerank(items: List[Any], top_k: int) -> bool:
        """Use an LLM reranker only when vector ordering is genuinely ambiguous."""
        scores = [
            float(item.get("score", 0.0))
            for item in items
            if isinstance(item, dict)
        ]
        if len(scores) <= top_k:
            return False
        boundary_margin = scores[top_k - 1] - scores[top_k]
        threshold = float(os.getenv("RAG_RERANK_MIN_MARGIN", "0.025"))
        return boundary_margin < threshold

    async def search_optimized(
        self,
        tool_name: str,
        query: str,
        top_k: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Prefer one direct retrieval and expand only when confidence is low."""
        recall_k = max(top_k + 2, 5)
        direct = await self.call(
            tool_name,
            {"query": query, "top_k": recall_k},
            context,
            use_cache=True,
        )
        if not direct.success or not isinstance(direct.data, list):
            return direct

        items = direct.data
        if len(items) <= top_k:
            direct.data = items[:top_k]
            return direct

        scores = [
            float(item.get("score", 0.0))
            for item in items
            if isinstance(item, dict)
        ]
        top_score = scores[0] if scores else 0.0
        margin = top_score - scores[top_k] if len(scores) > top_k else top_score
        min_score = float(os.getenv("RAG_MIN_DIRECT_SCORE", "0.25"))
        min_margin = float(os.getenv("RAG_MIN_DIRECT_MARGIN", "0.03"))
        mode = os.getenv("RAG_REWRITE_MODE", "adaptive").lower()

        # Clear, sufficiently relevant results keep the inexpensive vector order.
        clear_query = len(re.findall(r"\w+", query, flags=re.UNICODE)) >= 4
        if mode == "never" or (
            mode == "adaptive"
            and top_score >= min_score
            and (margin >= min_margin or clear_query)
        ):
            direct.data = items[:top_k]
            return direct

        # Ambiguous or low-confidence retrieval gets the full recall pipeline.
        return await self.search_with_rewrite(tool_name, query, top_k=top_k, context=context)

    # ── Result reranking to improve relevance ─────────────────────────────────

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """Use the LLM to rerank retrieved results by semantic usefulness."""
        if len(items) <= top_k:
            return items

        # Serialize results as text for LLM scoring
        items_text = "\n".join(f"{i}. {json.dumps(item, ensure_ascii=False)[:200]}"
                               for i, item in enumerate(items))
        prompt = f"""Score the following retrieval results for relevance to the user query from 0 to 10, then return a JSON array.
User query: "{query}"
Retrieval results:
{items_text}

Return format (indices ordered from most to least relevant): [most_relevant_index, ..., least_relevant_index]
Return only the JSON array."""
        prompt = self._clean_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            order: List[int] = json.loads(raw[s:e])
            reranked = [items[i] for i in order if 0 <= i < len(items)]
            return reranked[:top_k]
        except Exception as ex:
            logger.warning(f"Reranking failed; returning the original order: {ex}")
            return items[:top_k]

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict, rerank_top_k: int = 0) -> str:
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict, rerank_top_k: int = 0) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(name, params, rerank_top_k)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
    ) -> None:
        if len(self._cache) >= 5000:
            # Remove the oldest quarter
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params, rerank_top_k)] = (data, time.monotonic() + ttl, reranked)

    # ── Argument validation ────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """Validate arguments against a tool's JSON Schema."""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"Tool {tool.name} is missing required parameter: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"Tool {tool.name} parameter {key} has the wrong type: expected {expected_type}, got {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Remove Unicode surrogate characters before LLM request encoding."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
