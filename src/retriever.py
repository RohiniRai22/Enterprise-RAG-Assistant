
from src.embeddings import EmbeddingEngine
from src.vector_store import FAISSVectorStore


class Retriever:

    def __init__(self, storage_path="vector_store"):

        self.embedding_engine = EmbeddingEngine()

        self.vector_store = FAISSVectorStore(
            embedding_dimension=self.embedding_engine.embedding_dimension,
            storage_path=storage_path
        )

        self.vector_store.load()

    def retrieve(
        self,
        query,
        top_k=3,
        department=None,
        min_score=None
    ):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if min_score is not None:
            if (
                not isinstance(min_score, (int, float))
                or isinstance(min_score, bool)
                or not -1 <= min_score <= 1
            ):
                raise ValueError("min_score must be between -1 and 1.")

        # STEP 1: Generate query embedding
        query_embedding = self.embedding_engine.generate_query_embedding(
            query
        )

        # STEP 2: Check FAISS index
        total_chunks = self.vector_store.index.ntotal

        if total_chunks == 0:
            return []

        # STEP 3: Search all chunks
        results = self.vector_store.search(
            query_embedding,
            top_k=total_chunks
        )

        # STEP 4: Apply department filter
        if department:
            results = [
                result for result in results
                if result["document"]["metadata"].get(
                    "department", ""
                ).strip().lower() == department.strip().lower()
            ]

        # STEP 5: Apply minimum similarity score
        if min_score is not None:
            results = [
                result for result in results
                if result["score"] >= min_score
            ]

        # STEP 6: Return top relevant results
        return results[:top_k]
