"""Multi-turn memory using Redis working memory and ChromaDB episodic memory and profiles."""
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis
from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """Complete memory context passed to an agent."""
    recent_messages:  List[Message]   # Working memory: recent conversation
    relevant_history: List[str]       # Episodic memory: semantically related history
    user_profile:     Dict[str, Any]  # User profile: preferences and common entities
    summary:          str             # Compressed summary of the current conversation

    @staticmethod
    def _clean(text: str) -> str:
        """Remove Unicode surrogate characters to prevent encoding errors."""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """Format memory context for LLM consumption."""
        parts = []
        if self.summary:
            parts.append(f"[Conversation summary]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[Relevant history]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[User profile]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[Recent conversation]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """Manage Redis working memory and persistent ChromaDB memory and profiles."""

    WORKING_MAX   = 20    # Compress after the working-memory maximum is exceeded
    COMPRESS_AT   = 15    # Retain the summary and five latest messages after compression
    HISTORY_TOP_K = 5     # Number of episodic-memory results

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-sonnet-4-6",
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model

        self._redis = redis.from_url(redis_url, decode_responses=True)

        # Prefer standalone ChromaDB in Docker Compose and fall back to local embedded mode.
        try:
            # Explicitly disable ChromaDB telemetry to avoid PostHog compatibility errors.
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # Test the connection
            logger.info(f"ChromaDB connected: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB unavailable; using local embedded mode: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
        self._chroma = chroma

        # Episodic memory stores historical conversation segments
        self._episodic = chroma.get_or_create_collection("episodic")
        # User profiles store extracted preferences and entities
        self._profile  = chroma.get_or_create_collection("user_profile")

    def health_status(self) -> Dict[str, bool]:
        """Report memory dependency health for /health and frontend status."""
        status = {"redis": False, "chromadb": False}
        try:
            status["redis"] = bool(self._redis.ping())
        except Exception as ex:
            logger.warning(f"Redis health check failed: {ex}")
        try:
            status["chromadb"] = self._chroma.heartbeat() is not None
        except Exception as ex:
            logger.warning(f"ChromaDB health check failed: {ex}")
        return status

    # ── Write operations ───────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a message to working memory and compress after the threshold."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # Prepend to the Redis list so the newest item is first
        self._redis.lpush(key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        self._redis.expire(key, 86400)  # 24h TTL

        # Trigger compression after exceeding the threshold
        if self._redis.llen(key) >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """Extract preferences from working memory and persist the updated user profile."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        prompt = f"""Extract user preferences and key entities from the following conversation and return JSON.
Conversation:
{text}

Return format: {{"preferences": ["..."], "entities": {{"policy_domains": [], "request_types": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[s:e])

            doc_id = f"{user_id}_profile_{conv_id}"
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            try:
                self._profile.delete(ids=[doc_id])
            except Exception:
                pass

            # Pass documents directly so ChromaDB generates embeddings without Voyage API.
            self._profile.add(
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat()}],
            )
            logger.info(f"User profile updated: {user_id}")
        except Exception as ex:
            logger.warning(f"User-profile update failed: {ex}")

    # ── Read operations ────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """Build complete context and retrieve semantically relevant episodic history."""
        # 1. Working memory with recent messages from the current conversation
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. Episodic memory retrieved semantically across conversations
        history = await self._search_episodic(user_id, query or (recent[-1].content if recent else ""))

        # 3. User profile
        profile = await self._get_profile(user_id)

        # 4. Conversation summary when compression has occurred
        summary = self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── Compression to control context growth ──────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """Compress old messages into a Redis summary and ChromaDB episodic memory."""
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # Keep the five most recent messages
        keep        = messages[-5:]

        # LLM summary
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"Summarize the key information in this conversation in two or three sentences:\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
        except Exception:
            summary = f"The conversation contains {len(to_compress)} messages (summary generation failed)."

        # Store the summary in Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = self._redis.get(skey) or ""
        new_summary = self._safe_text(f"{old_summary}\n{summary}").strip()
        self._redis.setex(skey, 86400, new_summary)

        # Store old messages in episodic memory
        await self._store_episodic(user_id, conv_id, text, summary)

        # Reset working memory to the five most recent messages
        key = self._wm_key(user_id, conv_id)
        self._redis.delete(key)
        for m in reversed(keep):
            self._redis.lpush(key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        self._redis.expire(key, 86400)
        logger.info(f"Working-memory compression completed: {user_id}/{conv_id}, summary length {len(summary)}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush stores newest first; reverse to restore order
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """Retrieve episodic memory using ChromaDB's built-in embeddings."""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            # Pass query_texts directly so ChromaDB generates vectors for matching.
            results = self._episodic.query(
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [self._safe_text(doc) for doc in docs if isinstance(doc, str) and doc.strip()]
        except Exception as ex:
            logger.warning(f"Episodic-memory retrieval failed: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """Store compressed conversation segments in episodic memory."""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # Pass documents directly so ChromaDB generates embeddings.
            self._episodic.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
        except Exception as ex:
            logger.warning(f"Episodic-memory storage failed: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """Return the latest user profile."""
        try:
            results = self._profile.get(where={"user_id": user_id}, limit=1)
            if results["documents"]:
                return json.loads(results["documents"][0])
        except Exception:
            pass
        return {}

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """Convert a value to a plain UTF-8 string accepted by ChromaDB."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """Recursively sanitize metadata for safe Redis and ChromaDB I/O."""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
