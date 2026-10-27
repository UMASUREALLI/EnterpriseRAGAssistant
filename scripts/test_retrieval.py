
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Connect to the database created during ingestion.
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_collection(
        name="company_documents"
    )

    # Load the same embedding model used for document chunks.
    model = SentenceTransformer(MODEL_NAME)

    question = "Who needs to approve my time off?"

    # Convert the question into a vector.
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    # Search for the most relevant document chunks.
    results = collection.query(
        query_embeddings=query_vector,
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nQuestion: {question}\n")

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print(f"Result {i + 1}")
        print(f"Source: {metadata['source']}")
        print(f"Distance: {distance:.4f}")
        print(f"Content: {document}\n")


if __name__ == "__main__":
    main()
