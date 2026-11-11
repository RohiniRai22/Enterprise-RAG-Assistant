
from src.retriever import Retriever
from src.gemini_generator import GeminiGenerator


class RAGPipeline:

    def __init__(self):

        print("Initializing RAG pipeline...")

        # Initialize retrieval component
        self.retriever = Retriever()

        # Initialize Gemini generation component
        self.generator = GeminiGenerator()

        print("RAG pipeline initialized successfully!")

    def ask(
        self,
        question,
        top_k=3,
        department=None,
        min_score=None
    ):

        """
        Complete RAG workflow:
        Question -> Retrieval -> Gemini -> Answer
        """

        # Step 1: Retrieve relevant document chunks
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            department=department,
            min_score=min_score
        )

        # Step 2: Generate answer using retrieved context
        answer = self.generator.generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        # Step 3: Prepare source information
        sources = []

        for result in retrieved_chunks:

            document = result["document"]
            metadata = document["metadata"]

            source_info = {
                "source": metadata.get("source"),
                "department": metadata.get("department"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "similarity_score": round(
                    result["score"], 4
                )
            }

            # Avoid duplicate source entries
            if source_info not in sources:
                sources.append(source_info)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_chunks
        }
