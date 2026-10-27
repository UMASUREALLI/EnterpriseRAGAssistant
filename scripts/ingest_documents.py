
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from app.ingestion.chunking import split_text
from app.ingestion.document_loader import (
    SUPPORTED_EXTENSIONS,
    load_document,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "company_documents"


def make_chunk_id(filename, chunk_index):
    """Create a stable ID for each file's chunk."""
    filename_hash = hashlib.sha256(
        filename.encode("utf-8")
    ).hexdigest()[:16]

    return f"{filename_hash}-{chunk_index}"


def main():
    # 1. Find supported files.
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found: {DOCUMENTS_DIR}"
        )

    document_files = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not document_files:
        raise FileNotFoundError(
            f"No supported documents found in {DOCUMENTS_DIR}"
        )

    # 2. Connect to the existing vector database.
    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 3. Load the embedding model once.
    print("Loading the embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    successful_documents = 0
    total_chunks_added = 0

    # 4. Process each document independently.
    for file_path in document_files:
        print(f"\nProcessing: {file_path.name}")

        try:
            text = load_document(file_path)
            chunks = split_text(text)

            if not chunks:
                print("Skipped: no text chunks were created.")
                continue

            # Generate embeddings before modifying existing records.
            embeddings = model.encode(
                chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            ids = [
                make_chunk_id(file_path.name, index)
                for index in range(len(chunks))
            ]

            metadatas = [
                {
                    "source": file_path.name,
                    "chunk_index": index,
                }
                for index in range(len(chunks))
            ]

            # Replace only this file's old chunks.
            # Other documents remain in the collection.
            collection.delete(
                where={"source": file_path.name}
            )

            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            successful_documents += 1
            total_chunks_added += len(chunks)

            print(f"Indexed: {len(chunks)} chunks")

        except Exception as error:
            print(f"Failed to process {file_path.name}: {error}")
            print("Other documents will still be processed.")

    print("\n" + "=" * 40)
    print("INGESTION SUMMARY")
    print("=" * 40)
    print(f"Documents successfully processed: {successful_documents}")
    print(f"Chunks indexed in this run: {total_chunks_added}")
    print(f"Total chunks in database: {collection.count()}")
    print(f"Database location: {DATABASE_DIR}")


if __name__ == "__main__":
    main()
