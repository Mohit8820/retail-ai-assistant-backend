import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class PineconeService:

    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX")

        self.create_index_if_not_exists()

        self.index = self.pc.Index(self.index_name)

    def create_index_if_not_exists(self):

        existing_indexes = self.pc.list_indexes().names()

        if self.index_name not in existing_indexes:

            self.pc.create_index(
                name=self.index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            print("Pinecone index created")

        else:
            print("Pinecone index already exists")


if __name__ == "__main__":
    pinecone_service = PineconeService()