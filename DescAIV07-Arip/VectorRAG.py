from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

def search_batch(batch_cols, metadf):
    docs = metadf.apply(lambda row: str(row['column_name']).lower(), axis=1).tolist()

    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    embeddings = model.encode(docs)

    client_qdrant = QdrantClient(host="localhost", port=6333)

    client_qdrant.recreate_collection(
        collection_name="my_collection",
        vectors_config={"size": len(embeddings[0]), "distance": "Cosine"}
    )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"definition": str(metadf.iloc[i]['definition'])})
        for i in range(len(embeddings))
    ]
    client_qdrant.upsert(collection_name="my_collection", points=points)

    results = []
    for col in batch_cols:
        q_vector = model.encode(col.lower()).tolist()
        res = client_qdrant.search(
            collection_name="my_collection",
            query_vector=q_vector,
            limit=1
        )
        if res and res[0].score > 0.7:
            results.append(res[0].payload["definition"])
        else:
            results.append(f"- (No info found for '{col}' on RAG context)")
        
    context = "\n".join(results)
    print("=== Context for LLM ===")
    print(batch_cols)
    print(context)

    return context