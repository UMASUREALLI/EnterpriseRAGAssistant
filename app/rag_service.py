
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.retrieval.reranker import DocumentReranker
from app.generation.llm import KnowledgeAssistantLLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "company_documents"


class RAGService:
    """Runs retrieval, reranking, and answer generation."""

    def __init__(self):
        print("Loading the RAG service...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(DATABASE_PATH)
        )

        self.collection = self.chroma_client.get_collection(
            COLLECTION_NAME
        )

        if self.collection.count() == 0:
            raise RuntimeError(
                "The document collection is empty. "
                "Run python -m scripts.ingest_documents first."
            )

        self.reranker = DocumentReranker()
        self.llm = KnowledgeAssistantLLM()

        print(
            f"RAG service ready: "
            f"{self.collection.count()} chunks indexed."
        )

    def ask(self, question: str) -> dict:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        # 1. Embed the user's question.
        query_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
        ).tolist()

        # 2. Retrieve more candidates for reranking.
        result_count = min(10, self.collection.count())

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=result_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = []
        seen_chunks = set()

        retrieved_texts = results.get("documents", [[]])[0] or []
        retrieved_metadata = results.get("metadatas", [[]])[0] or []
        retrieved_distances = results.get("distances", [[]])[0] or []

        for text, metadata, distance in zip(
            retrieved_texts,
            retrieved_metadata,
            retrieved_distances,
        ):
            if not text or not metadata:
                continue

            source = metadata.get("source", "Unknown source")
            chunk_index = metadata.get("chunk_index", -1)

            # Prevent the same source chunk from being used twice.
            chunk_key = (source, chunk_index)

            if chunk_key in seen_chunks:
                continue

            seen_chunks.add(chunk_key)

            documents.append(
                {
                    "text": text,
                    "source": source,
                    "chunk_index": chunk_index,
                    "vector_distance": float(distance),
                }
            )

        if not documents:
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the available documents."
                ),
                "sources": [],
            }

        # 3. Rerank the unique candidates.
        ranked_documents = self.reranker.rerank(
            question,
            documents,
            top_k=min(3, len(documents)),
        )

        # Defensive deduplication in case the reranker changes order.
        unique_ranked = []
        seen_ranked = set()

        for document in ranked_documents:
            key = (
                document.get("source"),
                document.get("chunk_index"),
            )

            if key in seen_ranked:
                continue

            seen_ranked.add(key)
            unique_ranked.append(document)

        print(
            f"Question: {question}\n"
            f"Unique retrieved chunks: {len(documents)}\n"
            f"Chunks passed to LLM: {len(unique_ranked)}"
        )

        # 4. Generate the grounded answer.
        answer = self.llm.generate_answer(
            question,
            unique_ranked,
        )

        if not isinstance(answer, str) or not answer.strip():
            answer = (
                "I couldn't generate an answer this time. "
                "Please try again."
            )

        # 5. Return unique source references.
        sources = [
            {
                "source": document.get("source", "Unknown source"),
                "chunk_index": document.get("chunk_index", -1),
                "rerank_score": document.get("rerank_score"),
            }
            for document in unique_ranked
        ]

        return {
            "question": question,
            "answer": answer.strip(),
            "sources": sources,
        }
