
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.retrieval.reranker import DocumentReranker


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_questions.json"
DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db"

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "company_documents"
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3


def is_expected_chunk(result, expected):
    """Check whether a result matches the expected document chunk."""
    metadata = result.get("metadata") or {}

    return (
        metadata.get("source") == expected["expected_source"]
        and metadata.get("chunk_index") == expected["expected_chunk_index"]
    )


def reciprocal_rank(results, expected):
    """Return 1/rank for the first correct result, or 0 if missing."""
    for rank, result in enumerate(results, start=1):
        if is_expected_chunk(result, expected):
            return 1 / rank
    return 0.0


def calculate_metrics(all_results, k):
    """Calculate Recall@K and MRR for all evaluation questions."""
    if not all_results:
        return 0.0, 0.0

    hits = 0
    reciprocal_ranks = []

    for item in all_results:
        top_k_results = item["results"][:k]

        if any(
            is_expected_chunk(result, item["expected"])
            for result in top_k_results
        ):
            hits += 1

        reciprocal_ranks.append(
            reciprocal_rank(item["results"], item["expected"])
        )

    recall_at_k = hits / len(all_results)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return recall_at_k, mrr


def main():
    # Load evaluation questions
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("The evaluation dataset is empty.")

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=str(DATABASE_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Loading reranker...")
    reranker = DocumentReranker()

    vector_results = []
    reranked_results = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]

        print(f"\n{number}. Question: {question}")

        # Convert the question into an embedding
        query_embedding = embedding_model.encode(
            question
        ).tolist()

        # Retrieve the top 5 chunks from ChromaDB
        retrieved = collection.query(
            query_embeddings=[query_embedding],
            n_results=RETRIEVAL_TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        documents = []

        for index, text in enumerate(retrieved["documents"][0]):
            metadata = retrieved["metadatas"][0][index] or {}
            distance = retrieved["distances"][0][index]

            documents.append({
                "text": text,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "distance": float(distance),
            })

        # Save the original vector search ranking
        vector_results.append({
            "expected": item,
            "results": [
                {
                    "metadata": {
                        "source": doc["source"],
                        "chunk_index": doc["chunk_index"],
                    }
                }
                for doc in documents
            ],
        })

        # Rerank the retrieved chunks with the CrossEncoder
        reranked = reranker.rerank(
            question,
            documents,
            top_k=RERANK_TOP_K,
        )

        reranked_results.append({
            "expected": item,
            "results": [
                {
                    "metadata": {
                        "source": doc["source"],
                        "chunk_index": doc["chunk_index"],
                    }
                }
                for doc in reranked
            ],
        })

        # Show where the expected chunk appears
        vector_rank = next(
            (
                rank
                for rank, doc in enumerate(documents, start=1)
                if (
                    doc["source"] == item["expected_source"]
                    and doc["chunk_index"] == item["expected_chunk_index"]
                )
            ),
            None,
        )

        reranked_rank = next(
            (
                rank
                for rank, doc in enumerate(reranked, start=1)
                if (
                    doc["source"] == item["expected_source"]
                    and doc["chunk_index"] == item["expected_chunk_index"]
                )
            ),
            None,
        )

        print(f"  Correct chunk in vector results: rank {vector_rank or 'not found'}")
        print(f"  Correct chunk after reranking:   rank {reranked_rank or 'not found'}")

    # Calculate metrics
    vector_recall, vector_mrr = calculate_metrics(
        vector_results, RETRIEVAL_TOP_K
    )
    reranked_recall, reranked_mrr = calculate_metrics(
        reranked_results, RERANK_TOP_K
    )

    print("\n" + "=" * 48)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 48)
    print(f"Evaluation questions: {len(questions)}")

    print("\nVector retrieval:")
    print(f"  Recall@{RETRIEVAL_TOP_K}: {vector_recall:.2%}")
    print(f"  MRR:       {vector_mrr:.3f}")

    print("\nAfter reranking:")
    print(f"  Recall@{RERANK_TOP_K}: {reranked_recall:.2%}")
    print(f"  MRR:       {reranked_mrr:.3f}")

    print("\nNote: These results use a small starter dataset.")
    print("Add more varied questions before drawing strong conclusions.")


if __name__ == "__main__":
    main()
