
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class DocumentReranker:
    def __init__(self):
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, question, documents, top_k=3):
        if not documents:
            return []

        pairs = [
            (question, document["text"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_documents = []

        for document, score in zip(documents, scores):
            ranked_documents.append({
                **document,
                "rerank_score": float(score),
            })

        ranked_documents.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked_documents[:top_k]
