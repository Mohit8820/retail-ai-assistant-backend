import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

class EmbeddingService:

    def __init__(self):
        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )
        self.model = "gemini-embedding-001"

    def generate_embedding(self, text: str):

        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        return response.embeddings[0].values


if __name__ == "__main__":

    emb = EmbeddingService()

    vector = emb.generate_embedding(
        "hitmo"
    )

    print(len(vector))