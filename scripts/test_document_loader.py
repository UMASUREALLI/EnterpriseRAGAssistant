
from pathlib import Path

from app.ingestion.document_loader import load_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main():
    supported_files = [
        path
        for path in sorted(DOCUMENTS_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in {
            ".txt", ".pdf", ".docx"
        }
    ]

    if not supported_files:
        print(f"No supported documents found in {DOCUMENTS_DIR}")
        return

    for file_path in supported_files:
        print(f"\nTesting: {file_path.name}")

        try:
            text = load_document(file_path)
            print(f"Extracted characters: {len(text)}")
            print(f"Preview: {text[:200]}")
        except Exception as error:
            print(f"Failed: {error}")


if __name__ == "__main__":
    main()
