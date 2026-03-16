import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from services.pinecone_client import PineconeService
from services.embedding_service import EmbeddingService
from services.gemini_client import GeminiClient


class RetailRAGService:

    def __init__(self):

        self.pinecone = PineconeService()
        self.embedding = EmbeddingService()
        self.llm = GeminiClient()

    def ask(self, question: str):

        # 1️⃣ Convert question → vector
        query_vector = self.embedding.generate_embedding(question)

        # 2️⃣ Search Pinecone
        results = self.pinecone.query_vectors(query_vector, top_k=5)

        # 3️⃣ Extract insight text
        context = "\n".join(
            match["metadata"]["text"]
            for match in results["matches"]
        )

        # 4️⃣ Prompt LLM
        prompt = f"""
You are a retail analytics AI assistant.

Use the following insights to answer the question.

Insights:
{context}

Question:
{question}

Answer clearly and explain the reasoning.
"""

        answer = self.llm.generate(prompt)

        return answer