
from pathlib import Path

import pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def load_document(file_path: str | Path) -> str:
    """
    Extract text from a TXT, PDF, or DOCX file.

    Returns:
        Extracted text as a string.

    Raises:
        ValueError: Unsupported file type or empty extracted text.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            "Supported types are TXT, PDF, and DOCX."
        )

    if extension == ".txt":
        text = path.read_text(encoding="utf-8-sig")

    elif extension == ".pdf":
        with pymupdf.open(path) as pdf:
            text = "\n".join(
                page.get_text() for page in pdf
            )

    elif extension == ".docx":
        document = Document(path)
        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]
        text = "\n".join(paragraphs)

    text = text.strip()

    if not text:
        raise ValueError(
            f"No text could be extracted from {path.name}. "
            "The file may be empty, scanned, or image-only."
        )

    return text
