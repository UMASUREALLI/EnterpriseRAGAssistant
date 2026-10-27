
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.retrieval.reranker import DocumentReranker


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("company_documents")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    question = "Who needs to approve my time off?"

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(3, collection.count()),
        include=["documents", "metadatas"],
    )

    documents = []

    for text, metadata in zip(
        results["documents"][0],
        results["metadatas"][0],
    ):
        documents.append({
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
        })

    print("\nBefore re-ranking:")
    for document in documents:
        print("-", document["text"])

    reranker = DocumentReranker()
    ranked = reranker.rerank(
        question,
        documents,
        top_k=3,
    )

    print("\nAfter re-ranking:")
    for item in ranked:
        print(f"Score: {item['rerank_score']:.4f}")
        print(f"Source: {item['source']}")
        print(f"Text: {item['text']}\n")


if __name__ == "__main__":
    main()
