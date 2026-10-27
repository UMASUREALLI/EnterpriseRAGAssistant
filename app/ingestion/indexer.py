
import hashlib
from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.ingestion.chunking import split_text
from app.ingestion.document_loader import load_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "company_documents"


def calculate_content_hash(text: str) -> str:
    """Create a stable hash from extracted document text."""
    normalized_text = text.replace("\r\n", "\n").strip()

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


class DocumentIndexer:
    def __init__(self):
        print("Initializing document indexer...")

        self.model = SentenceTransformer(MODEL_NAME)

        client = chromadb.PersistentClient(
            path=str(DATABASE_DIR)
        )

        self.collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _find_duplicate(
        self,
        content_hash: str,
        chunks: list[str],
    ) -> bool:
        """Check hashed records and legacy records without deleting data."""

        # First, check records indexed with content_hash.
        hashed_records = self.collection.get(
            where={"content_hash": content_hash},
            include=["metadatas"],
            limit=1,
        )

        if hashed_records["ids"]:
            return True

        # Next, inspect legacy records that have no content_hash.
        # This helps detect exact matches from earlier indexing runs.
        records = self.collection.get(
            include=["documents", "metadatas"],
        )

        legacy_groups = defaultdict(list)

        for document, metadata in zip(
            records["documents"],
            records["metadatas"],
        ):
            metadata = metadata or {}

            if metadata.get("content_hash"):
                continue

            source = metadata.get("source")
            if not source or document is None:
                continue

            # Group uploaded records by document ID when available.
            # Older batch-ingested records may have only a source name.
            group_key = (
                source,
                metadata.get("document_id") or "legacy",
            )

            legacy_groups[group_key].append(
                (
                    metadata.get("chunk_index", 0),
                    document,
                )
            )

        # Compare complete, ordered chunk sequences.
        for records_for_document in legacy_groups.values():
            ordered_chunks = [
                document
                for _, document in sorted(
                    records_for_document,
                    key=lambda item: item[0],
                )
            ]

            if ordered_chunks == chunks:
                return True

        return False

    def index_document(
        self,
        file_path: str | Path,
        original_filename: str,
        document_id: str,
    ):
        """Index new content without deleting existing records."""

        text = load_document(file_path)
        content_hash = calculate_content_hash(text)

        chunks = split_text(text)

        if not chunks:
            raise ValueError("No text chunks were created.")

        # Do not add another copy of identical content.
        if self._find_duplicate(content_hash, chunks):
            return {
                "filename": original_filename,
                "chunks_indexed": 0,
                "total_chunks": self.collection.count(),
                "message": "Identical content is already indexed.",
                "duplicate": True,
            }

        embeddings = self.model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        ids = [
            f"{document_id}-{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "source": original_filename,
                "document_id": document_id,
                "chunk_index": index,
                "content_hash": content_hash,
            }
            for index in range(len(chunks))
        ]

        # Use unique IDs for this upload.
        # No existing records are deleted.
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "filename": original_filename,
            "chunks_indexed": len(chunks),
            "total_chunks": self.collection.count(),
            "message": "Document indexed successfully.",
            "duplicate": False,
        }
