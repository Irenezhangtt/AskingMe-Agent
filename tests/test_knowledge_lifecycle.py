"""Regression tests for knowledge-document lifecycle management."""
import unittest

from mcp.knowledge_base import DuplicateDocumentError, KnowledgeBase


class FakeCollection:
    """Small in-memory subset of the Chroma collection API used by lifecycle tests."""

    def __init__(self):
        self.rows = {}

    def add(self, ids, documents, metadatas):
        for row_id, document, metadata in zip(ids, documents, metadatas):
            self.rows[row_id] = {"document": document, "metadata": dict(metadata)}

    def get(self, where=None, include=None):
        matches = []
        for row_id, row in self.rows.items():
            if where and any(row["metadata"].get(key) != value for key, value in where.items()):
                continue
            matches.append((row_id, row))
        return {
            "ids": [row_id for row_id, _ in matches],
            "documents": [row["document"] for _, row in matches],
            "metadatas": [dict(row["metadata"]) for _, row in matches],
        }

    def update(self, ids, metadatas):
        for row_id, metadata in zip(ids, metadatas):
            self.rows[row_id]["metadata"] = dict(metadata)

    def delete(self, ids):
        for row_id in ids:
            self.rows.pop(row_id, None)

    def count(self):
        return len(self.rows)


class KnowledgeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.__new__(KnowledgeBase)
        self.kb._collection = FakeCollection()

    def test_duplicate_content_is_rejected(self):
        first = self.kb.import_document(
            title="Travel Policy",
            content="Employees require manager approval before travel.",
            status="draft",
        )
        with self.assertRaises(DuplicateDocumentError) as raised:
            self.kb.import_document(
                title="Copied Travel Policy",
                content="Employees   require manager approval before travel.",
            )
        self.assertEqual(raised.exception.document["document_id"], first["document_id"])

    def test_approval_archives_the_earlier_version(self):
        first = self.kb.import_document(
            title="Travel Policy (1.0)",
            content="Version one policy text.",
            version="1.0",
            status="approved",
        )
        second = self.kb.import_document(
            title="Travel Policy (2.0)",
            content="Version two policy text.",
            version="2.0",
            status="draft",
            replaces_document_id=first["document_id"],
        )

        self.kb.approve_document(second["document_id"])

        self.assertEqual(self.kb.get_document(first["document_id"])["status"], "archived")
        self.assertEqual(self.kb.get_document(second["document_id"])["status"], "approved")

    def test_delete_removes_all_document_chunks(self):
        document = self.kb.import_document(
            title="Long Policy",
            content=("A complete policy sentence. " * 80),
            status="draft",
        )
        self.assertGreater(document["added_chunks"], 1)

        deleted = self.kb.delete_document(document["document_id"])

        self.assertEqual(deleted["document_id"], document["document_id"])
        self.assertEqual(self.kb.doc_count, 0)


if __name__ == "__main__":
    unittest.main()
