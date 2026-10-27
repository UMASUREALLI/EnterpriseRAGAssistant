
import re


def split_text(text, chunk_size=450, overlap=80):
    """Split text at sentence boundaries with approximate overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = " ".join(text.split())

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])

        if current and len(candidate) > chunk_size:
            chunks.append(" ".join(current))

            # Preserve recent complete sentences as overlap.
            previous = []
            for item in reversed(current):
                candidate_overlap = " ".join([item] + previous)

                if len(candidate_overlap) > overlap:
                    break

                previous.insert(0, item)

            current = previous

        current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks
