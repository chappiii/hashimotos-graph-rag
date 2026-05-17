from qdrant_client import QdrantClient

from retriever.config.ret_config import QDRANT_URL, TOP_K
from retriever.graph_ret.embed import embed_query


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    vector = embed_query(query)
    client = QdrantClient(url=QDRANT_URL)

    result = client.query_points(
        collection_name="chunks",
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": h.score, **h.payload} for h in result.points]
