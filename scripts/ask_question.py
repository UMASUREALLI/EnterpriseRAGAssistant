
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.retrieval.reranker import DocumentReranker
from app.generation.llm import KnowledgeAssistantLLM


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    question = input("Ask a company policy question: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # 1. Connect to the vector database.
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("company_documents")

    if collection.count() == 0:
        print("No documents are indexed yet.")
        return

    # 2. Convert the question into an embedding.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    # 3. Retrieve candidate passages.
    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(5, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        documents.append({
            "text": text,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "vector_distance": float(distance),
        })

    # 4. Re-rank passages for the question.
    reranker = DocumentReranker()

    ranked_documents = reranker.rerank(
        question,
        documents,
        top_k=3,
    )

    # 5. Generate an answer using the selected evidence.
    assistant = KnowledgeAssistantLLM()

    answer = assistant.generate_answer(
        question,
        ranked_documents,
    )

    print("\nAssistant answer:")
    print(answer)

    print("\nRetrieved sources:")
    for document in ranked_documents:
        print(
            f"- {document['source']}, "
            f"chunk {document['chunk_index']}"
        )


if __name__ == "__main__":
    main()
