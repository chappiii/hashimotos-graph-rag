import json
import uuid
import requests
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct

from kg_construction.config.kg_construction_config import (
    OLLAMA_URL,
    OLLAMA_MODEL,
    VECTOR_DIMENSION,
    EVIDENCE_MAPPING_PATH,
)


# Collection setup
def create_collection(client, collection_name, vector_dimension):
    try:
        client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        if "Not found: Collection" in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")
            return

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="parent_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )
        print("Payload index 'parent_id' created successfully.")
    except Exception as idx_err:
        if "already exists" in str(idx_err):
            print("Payload index 'parent_id' already exists, skipping.")
        else:
            print(f"Error creating payload index: {idx_err}")


# Embeddings
def ollama_embedding(text):
    try:
        if not text or text.strip() == "":
            return None

        print(f"Generating embedding: {text[:50]}...")

        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": text},
            timeout=60,
        )

        print(f"API response status: {response.status_code}")

        if response.status_code != 200:
            print(f"API error: {response.text}")
            return None

        response_data = response.json()
        embedding = response_data.get("embedding", [])
        print(f"Response received, embedding size: {len(embedding)}")
        return embedding

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama server. Is Ollama running?")
        return None
    except requests.exceptions.Timeout:
        print("Error: Embedding request timed out")
        return None
    except KeyError as e:
        print(f"Error: 'embedding' key not found in response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected embedding error: {e}")
        print(f"Error type: {type(e).__name__}")
        return None


# Parent ingestion
def ingest_parents(client, collection_name, metadata_dir):
    """Read grouped metadata (part-{n}.json), embed purpose_of_work, upload as parent docs."""
    metadata_path = Path(metadata_dir)
    files = sorted(metadata_path.glob("*.json"))

    if not files:
        print("No JSON files found in", metadata_dir)
        return

    points = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Invalid JSON, skipping: {fpath}")
                continue

        papers = data.get("papers", [])
        for paper in papers:
            parent_id_str = paper.get("paper_id")
            if parent_id_str is None:
                print(f"'paper_id' not found, skipping entry in: {fpath}")
                continue

            parent_id = int(parent_id_str)
            purpose_text = paper.get("purpose_of_work", "")

            vector = ollama_embedding(purpose_text)
            if vector is None:
                print(f"Could not generate embedding, using dummy vector: parent_id={parent_id}")
                vector = [0.0] * VECTOR_DIMENSION

            payload = {
                "parent_id": parent_id,
                "type": "parent",
                "doi": paper.get("doi"),
                "title": paper.get("title"),
                "published_year": paper.get("published_year"),
                "author_list": paper.get("author_list", []),
                "countries": paper.get("countries", []),
                "keywords": paper.get("keywords", []),
            }

            points.append(models.PointStruct(id=parent_id, vector=vector, payload=payload))

    if points:
        client.upsert(collection_name=collection_name, wait=True, points=points)
        print(f"Ingested {len(points)} parent records into '{collection_name}'.")
    else:
        print("No parents uploaded (no embeddings generated).")


# Children ingestion

def ingest_children(client, collection_name, root_dir, batch_size=20):
    """Walk gemini/{paper_id}/*_relations.json, embed evidence, upload as children."""
    root = Path(root_dir)
    seen_evidences = set()
    points = []
    counter = 0

    json_path = EVIDENCE_MAPPING_PATH
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        json_data = {}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    paper_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )

    for paper_dir in paper_dirs:
        parent_id = int(paper_dir.name) if paper_dir.name.isdigit() else None
        if parent_id is None:
            print(f"Skipping non-numeric folder: {paper_dir.name}")
            continue

        for file in paper_dir.iterdir():
            if not file.name.endswith("_relations.json"):
                continue

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            relations = data.get("relations", [])

            for entry in relations:
                evidence = entry.get("evidence", "").strip()
                if not evidence or evidence in seen_evidences:
                    continue
                seen_evidences.add(evidence)

                vector = ollama_embedding(evidence)
                if vector is None:
                    vector = [0.0] * VECTOR_DIMENSION
                elif len(vector) != VECTOR_DIMENSION:
                    if len(vector) > VECTOR_DIMENSION:
                        vector = vector[:VECTOR_DIMENSION]
                    else:
                        vector = vector + [0.0] * (VECTOR_DIMENSION - len(vector))

                # vector similarity dedup
                try:
                    search_res = client.query_points(
                        collection_name=collection_name, query=vector, limit=1,
                    )
                    if search_res.points and search_res.points[0].score > 0.9999:
                        existing_id = str(search_res.points[0].id)
                        json_data[existing_id] = evidence
                        continue
                except Exception as search_err:
                    print(f"Search error ignored: {type(search_err).__name__}: {search_err}")

                child_id = str(uuid.uuid4())
                payload = {"parent_id": parent_id, "evidence": evidence}

                points.append(PointStruct(id=child_id, vector=vector, payload=payload))
                json_data[child_id] = evidence
                counter += 1

                if len(points) >= batch_size:
                    try:
                        client.upsert(collection_name=collection_name, points=points, wait=True)
                        print(f"{counter} records uploaded (batch: {len(points)}).")
                        points.clear()
                    except Exception as upsert_err:
                        print(f"Upsert error: {type(upsert_err).__name__}: {upsert_err}")

                    try:
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)
                    except Exception as json_err:
                        print(f"JSON write error: {type(json_err).__name__}: {json_err}")

    if points:
        try:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            print(f"Last {len(points)} records uploaded (total: {counter}).")
        except Exception as upsert_err:
            print(f"Final upsert error: {type(upsert_err).__name__}: {upsert_err}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Total {counter} records processed and JSON updated.")
    except Exception as json_err:
        print(f"Final JSON write error: {type(json_err).__name__}: {json_err}")
