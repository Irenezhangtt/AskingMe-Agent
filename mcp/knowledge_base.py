"""ChromaDB-backed RAG knowledge base integrated as the knowledge_search MCP tool."""
import asyncio
import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """RAG knowledge base using ChromaDB's built-in embedding model."""

    COLLECTION_NAME = "askingme_policies_v2_en"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # Prefer the standalone ChromaDB service with server-side embeddings.
        self._use_server = False
        try:
            # Explicitly disable ChromaDB telemetry to avoid PostHog compatibility errors.
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"Knowledge-base ChromaDB connected: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB service unavailable; using local mode: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # Omit embedding_function for both server and local modes so ChromaDB handles it.
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "AskingMe Agent enterprise policy RAG knowledge base"},
        )

        # Import default documents when the knowledge base is empty
        if self._collection.count() == 0:
            self._load_default_docs()
        self._migrate_legacy_metadata()

    # ── Document management ───────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Dict[str, str]],
        *,
        status: str = "approved",
        version: str = "1.0",
        source_name: str = "",
    ) -> int:
        """Import documents while preserving the original immediate-approval API."""
        added = 0
        for document in documents:
            result = self.import_document(
                title=document.get("title", ""),
                content=document.get("content", ""),
                version=str(document.get("version", version)),
                status=str(document.get("status", status)),
                source_name=str(document.get("source_name", source_name)),
            )
            added += result["added_chunks"]
        return added

    def import_document(
        self,
        *,
        title: str,
        content: str,
        version: str = "1.0",
        status: str = "draft",
        source_name: str = "",
        replaces_document_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Import one versioned document with exact-content duplicate detection."""
        title = title.strip()
        content = content.strip()
        version = version.strip() or "1.0"
        status = status.strip().lower()
        if not title:
            raise ValueError("Document title is required")
        if not content:
            raise ValueError("Document content is empty")
        if status not in {"draft", "approved"}:
            raise ValueError("Status must be draft or approved")

        content_hash = self._content_hash(content)
        duplicate = self.find_duplicate(content_hash)
        if duplicate:
            raise DuplicateDocumentError(duplicate)

        document_id = str(uuid.uuid4())
        policy_key = self._policy_key(title)
        if replaces_document_id:
            try:
                policy_key = str(self.get_document(replaces_document_id).get("policy_key", policy_key))
            except KeyError as ex:
                raise ValueError("The document selected for replacement does not exist") from ex
        imported_at = datetime.now(timezone.utc).isoformat()
        chunks, chunk_metadata = self._prepare_chunks(content, document_id, title, metadata or {})
        ids = [f"{document_id}:{index}" for index in range(len(chunks))]
        metadatas = [
            {
                **chunk_metadata[index],
                "document_id": document_id,
                "title": title,
                "policy_key": policy_key,
                "version": version,
                "status": status,
                "content_hash": content_hash,
                "source_name": source_name or title,
                "replaces_document_id": replaces_document_id,
                "chunk_index": index,
                "total_chunks": len(chunks),
                "imported_at": imported_at,
                "approved_at": imported_at if status == "approved" else "",
            }
            for index in range(len(chunks))
        ]
        self._collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        if status == "approved":
            self._archive_other_versions(document_id, policy_key)
        logger.info("Imported document %s as %s (%s chunks)", document_id, status, len(ids))
        return {
            "document_id": document_id,
            "title": title,
            "version": version,
            "status": status,
            "content_hash": content_hash,
            "added_chunks": len(ids),
            "total_chunks": self.doc_count,
            "duplicate": False,
        }

    def find_duplicate(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Return the existing document with the same normalized content."""
        result = self._collection.get(
            where={"content_hash": content_hash},
            include=["metadatas"],
        )
        if not result.get("ids"):
            return None
        meta = result["metadatas"][0]
        return self._document_summary(meta)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Aggregate chunk metadata into document-version records."""
        result = self._collection.get(include=["metadatas"])
        documents: Dict[str, Dict[str, Any]] = {}
        for meta in result.get("metadatas") or []:
            document_id = str(meta.get("document_id", ""))
            if not document_id:
                continue
            if document_id not in documents:
                documents[document_id] = self._document_summary(meta)
        return sorted(
            documents.values(),
            key=lambda item: item.get("imported_at", ""),
            reverse=True,
        )

    def approve_document(self, document_id: str) -> Dict[str, Any]:
        """Approve one draft and archive other approved versions of its policy."""
        ids, metadatas = self._document_chunks(document_id)
        if not ids:
            raise KeyError(document_id)
        approved_at = datetime.now(timezone.utc).isoformat()
        updated = []
        for meta in metadatas:
            next_meta = dict(meta)
            next_meta["status"] = "approved"
            next_meta["approved_at"] = approved_at
            updated.append(next_meta)
        self._collection.update(ids=ids, metadatas=updated)
        self._archive_other_versions(document_id, str(updated[0].get("policy_key", "")))
        return self.get_document(document_id)

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete every vector chunk belonging to a document version."""
        record = self.get_document(document_id)
        ids, _ = self._document_chunks(document_id)
        if not ids:
            raise KeyError(document_id)
        self._collection.delete(ids=ids)
        return record

    def get_document(self, document_id: str) -> Dict[str, Any]:
        ids, metadatas = self._document_chunks(document_id)
        if not ids:
            raise KeyError(document_id)
        return self._document_summary(metadatas[0])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the document chunks most semantically relevant to a query."""
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"status": "approved"},
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # Convert ChromaDB distance to similarity
                    "chunk":    meta.get("chunk_index", 0),
                    "document_id": meta.get("document_id", ""),
                    "version": meta.get("version", ""),
                })

        return items

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP tool handler ──────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """Handle knowledge_search calls registered with MCPToolManager."""
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        # Chroma's Python client is synchronous; keep it off the FastAPI event loop.
        return await asyncio.to_thread(self.search, query, top_k)

    # ── Internal methods ───────────────────────────────────────────────────────

    def _document_chunks(self, document_id: str):
        result = self._collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        return result.get("ids") or [], result.get("metadatas") or []

    def _archive_other_versions(self, document_id: str, policy_key: str) -> None:
        if not policy_key:
            return
        result = self._collection.get(
            where={"policy_key": policy_key},
            include=["metadatas"],
        )
        ids_to_update, metas_to_update = [], []
        for chunk_id, meta in zip(result.get("ids") or [], result.get("metadatas") or []):
            if meta.get("document_id") == document_id or meta.get("status") != "approved":
                continue
            next_meta = dict(meta)
            next_meta["status"] = "archived"
            ids_to_update.append(chunk_id)
            metas_to_update.append(next_meta)
        if ids_to_update:
            self._collection.update(ids=ids_to_update, metadatas=metas_to_update)

    @staticmethod
    def _content_hash(content: str) -> str:
        normalized = re.sub(r"\s+", " ", content).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _policy_key(title: str) -> str:
        normalized = re.sub(r"\s+", " ", title).strip().lower()
        normalized = re.sub(r"\s*[\(\[]?(?:v(?:ersion)?\s*)?\d+(?:\.\d+)*[\)\]]?\s*$", "", normalized)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _document_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "document_id": meta.get("document_id", ""),
            "title": meta.get("title", ""),
            "policy_key": meta.get("policy_key", ""),
            "version": meta.get("version", "1.0"),
            "status": meta.get("status", "approved"),
            "content_hash": meta.get("content_hash", ""),
            "source_name": meta.get("source_name", ""),
            "replaces_document_id": meta.get("replaces_document_id", ""),
            "total_chunks": int(meta.get("total_chunks", 0)),
            "imported_at": meta.get("imported_at", ""),
            "approved_at": meta.get("approved_at", ""),
        }

    def _migrate_legacy_metadata(self) -> None:
        """Give pre-lifecycle chunks stable document IDs and approved status."""
        result = self._collection.get(include=["documents", "metadatas"])
        groups: Dict[str, List[tuple[str, str, Dict[str, Any]]]] = {}
        for chunk_id, text, meta in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            if meta.get("document_id"):
                continue
            groups.setdefault(str(meta.get("title", "Untitled document")), []).append(
                (chunk_id, text, meta)
            )
        for title, chunks in groups.items():
            chunks.sort(key=lambda item: int(item[2].get("chunk_index", 0)))
            content = "\n".join(item[1] for item in chunks)
            document_id = f"legacy-{hashlib.sha256(title.encode()).hexdigest()[:20]}"
            content_hash = self._content_hash(content)
            imported_at = datetime.now(timezone.utc).isoformat()
            ids, updated = [], []
            for chunk_id, _, meta in chunks:
                next_meta = dict(meta)
                next_meta.update({
                    "document_id": document_id,
                    "policy_key": self._policy_key(title),
                    "version": "legacy",
                    "status": "approved",
                    "content_hash": content_hash,
                    "source_name": "built-in",
                    "replaces_document_id": "",
                    "imported_at": imported_at,
                    "approved_at": imported_at,
                })
                ids.append(chunk_id)
                updated.append(next_meta)
            self._collection.update(ids=ids, metadatas=updated)

    def _prepare_chunks(self, content, document_id, title, metadata):
        chunks = self._chunk_text(content, chunk_size=500)
        return chunks, [dict(metadata) for _ in chunks]

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split long text by sentence or newline while respecting chunk_size."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # Split by sentence
        sentences = text.replace("\n", ". ").split(". ")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}. {sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """Import the default enterprise policy documents used by the demo."""
        default_docs = [
            {
                "title": "Travel and Expense Reimbursement Policy (2026)",
                "content": (
                    "This policy takes effect on January 1, 2026 and applies to all regular employees. "
                    "Before traveling, employees must submit a travel request in the internal workflow platform and obtain direct-manager approval. "
                    "The domestic hotel limit is CNY 600 per night in tier-one cities and CNY 450 per night elsewhere. "
                    "Intercity travel should use economy class or second-class rail; exceptions require written approval from the department head. "
                    "Expense claims must be submitted within 30 calendar days after the trip with valid receipts, approval records, and itinerary evidence. "
                    "Missing receipts require a no-receipt declaration jointly approved by the department head and Finance."
                ),
            },
            {
                "title": "Leave and Attendance Policy (2026)",
                "content": (
                    "Employees must submit leave requests through the HR system. "
                    "Annual leave should be requested at least three business days in advance and approved by the direct manager; five or more consecutive days also require department-head approval. "
                    "A one-day sick-leave request may be submitted the same day; two or more consecutive days require evidence from a recognized medical provider. "
                    "Personal leave should be requested one business day in advance; emergencies must be reported verbally and documented after return. "
                    "Missed punches must be corrected within three business days, with no more than three corrections per month. "
                    "Contact HR for cases involving pay deductions or employment disputes."
                ),
            },
            {
                "title": "System Account and Access Management Standard",
                "content": (
                    "Internal access follows least-privilege, job-based authorization, and periodic-review principles. "
                    "Baseline access for new employees is triggered by onboarding; business-system access requires an employee request. "
                    "Standard access requires direct-manager and system-owner approval; sensitive-data access also requires data-owner and information-security approval. "
                    "Access from a previous role must be removed within three business days after a transfer, and new access must be requested. "
                    "Departing employees' accounts are disabled on their effective termination date. Shared accounts, borrowed access, and approval bypasses are prohibited. "
                    "The agent explains procedures but cannot provision, modify, or remove access."
                ),
            },
            {
                "title": "Overtime and Compensatory Leave Policy",
                "content": (
                    "Overtime must be requested and approved by the direct manager in advance; emergency overtime may be documented the next day. "
                    "Less than one extra hour on a workday does not earn compensatory leave; approved time is accrued after the first hour. "
                    "Weekend and public-holiday overtime follows local law and the approved request. "
                    "Compensatory leave must be used within six months and should be requested at least one business day in advance. "
                    "Unapproved late work is not automatically treated as overtime."
                ),
            },
            {
                "title": "Procurement and Contract Approval Policy",
                "content": (
                    "A purchase of CNY 5,000 or less requires direct-manager approval. "
                    "Purchases above CNY 5,000 and up to CNY 50,000 also require department-head and Procurement approval. "
                    "Purchases above CNY 50,000 or involving long-term services require Procurement, Finance, and Legal review. "
                    "Supplier selection must retain competitive quotes or a sole-source justification; splitting purchases to avoid approval is prohibited. "
                    "Contracts require Legal review through the standard workflow before signature, and employees may not sign on the company's behalf without authority."
                ),
            },
            {
                "title": "Policy Version and Exception Handling Rules",
                "content": (
                    "Answers must use the latest effective version marked in the knowledge base. "
                    "When policies conflict, use the later effective provision whose scope applies. "
                    "When local rules, employment agreements, or specific notices override a general policy, employees should confirm with the responsible owner. "
                    "The agent must not approve exceptions or resolve missing or conflicting provisions on its own. "
                    "Items requiring approval must be submitted in the appropriate system, with the required approvers and supporting documents identified."
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"Imported {len(default_docs)} default knowledge-base documents")


class DuplicateDocumentError(ValueError):
    """Raised when normalized document content already exists."""

    def __init__(self, document: Dict[str, Any]):
        self.document = document
        super().__init__(
            f"Duplicate content already exists in '{document.get('title', 'Untitled')}' "
            f"(version {document.get('version', 'unknown')})"
        )
