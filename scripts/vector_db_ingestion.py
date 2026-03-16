import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from services.pinecone_client import PineconeService
from services.embedding_service import EmbeddingService



pinecone_service = PineconeService()
embedding_service = EmbeddingService()

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir,  "..", "data", "insights.json")) as f:
    insights = json.load(f)

vectors = []

for i, insight in enumerate(insights):

    embedding = embedding_service.generate_embedding(insight)

    vectors.append({
        "id": f"insight-{i}",
        "values": embedding,
        "metadata": {
            "text": insight
        }
    })

pinecone_service.upsert_vectors(vectors)

print("Insights stored in Pinecone")