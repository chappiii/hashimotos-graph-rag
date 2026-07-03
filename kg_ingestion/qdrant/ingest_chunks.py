"""
Embed chunk text (sliding window) and upsert into the `chunks` Qdrant collection.

Windowing:
  - References sections are skipped entirely (citation lists, not useful for retrieval)
  - Each chunk is split into 400-word windows with 80-word overlap
  - Each window = one Qdrant point
  - Point ID formula: paper_id * 1_000_000 + chunk_idx * 1_000 + window_idx

Skip logic: checks which point IDs already exist before embedding.

Run: uv run -m kg_ingestion.qdrant.ingest_chunks
"""

import re
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from kg_ingestion.config.kg_config import QDRANT_URL, CHUNKS_DIR, VECTOR_DIM
from kg_ingestion.qdrant.embed import embed
from extract_entity_relation.utils.file_utils import _should_skip, get_section_type

COLLECTION   = "chunks"
BATCH_SIZE   = 20
WINDOW_WORDS = 400
STRIDE_WORDS = 320  # WINDOW - OVERLAP (80-word overlap)


def sliding_windows(text: str) -> list[str]:
    """Split text into overlapping word windows. Returns at least one window."""
    words = text.split()
    if not words:
        return [""]
    windows = []
    start = 0
    while start < len(words):
        chunk = words[start : start + WINDOW_WORDS]
        windows.append(" ".join(chunk))
        if start + WINDOW_WORDS >= len(words):
            break
        start += STRIDE_WORDS
    return windows


def parse_chunk(filepath: Path) -> tuple[str, str]:
    """Return (header, body) from a chunk .md file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header = ""
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("Header:"):
            header = line.removeprefix("Header:").strip()
        if line.startswith("=" * 10):
            body_start = i + 1
            break
    body = "\n".join(lines[body_start:]).strip()
    return header, body


def collect_records(chunks_dir: Path) -> list[dict]:
    records = []
    skipped_refs = 0

    for paper_folder in sorted(chunks_dir.iterdir(), key=lambda p: int(p.name)):
        if not paper_folder.is_dir():
            continue
        paper_id = int(paper_folder.name)

        for chunk_file in sorted(paper_folder.glob("*.md")):
            m = re.match(r"^(\d+)", chunk_file.name)
            if not m:
                continue
            chunk_idx = int(m.group(1))
            header, body = parse_chunk(chunk_file)

            if _should_skip(chunk_file.name):
                skipped_refs += 1
                continue

            section_name = header or chunk_file.stem
            section_type = get_section_type(section_name)
            windows      = sliding_windows(body)

            for win_idx, win_text in enumerate(windows):
                point_id = paper_id * 1_000_000 + chunk_idx * 1_000 + win_idx
                records.append({
                    "point_id":     point_id,
                    "paper_id":     paper_id,
                    "section_name": section_name,
                    "section_type": section_type,
                    "chunk_filename": chunk_file.name,
                    "window_index": win_idx,
                    "window_count": len(windows),
                    "text":         win_text,
                })

    print(f"  skipped {skipped_refs} chunks (references/appendix)")
    return records


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it does not exist."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"  created collection '{COLLECTION}' (dim={VECTOR_DIM}, cosine)")


def existing_ids(client: QdrantClient, ids: list[int]) -> set[int]:
    found = client.retrieve(collection_name=COLLECTION, ids=ids, with_payload=False, with_vectors=False)
    return {p.id for p in found}


def ingest(client: QdrantClient, records: list[dict]) -> None:
    all_ids   = [r["point_id"] for r in records]
    skip_ids: set[int] = set()
    for i in range(0, len(all_ids), 500):
        skip_ids |= existing_ids(client, all_ids[i : i + 500])

    to_ingest = [r for r in records if r["point_id"] not in skip_ids]
    print(f"  {len(skip_ids)} already ingested, {len(to_ingest)} to embed")

    for i in range(0, len(to_ingest), BATCH_SIZE):
        batch = to_ingest[i : i + BATCH_SIZE]
        points = []
        for r in batch:
            vector = embed(r["text"])
            payload = {k: v for k, v in r.items() if k != "point_id"}
            points.append(PointStruct(id=r["point_id"], vector=vector, payload=payload))
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  upserted {min(i + BATCH_SIZE, len(to_ingest))}/{len(to_ingest)}")


def main() -> None:
    chunks_dir = Path(CHUNKS_DIR)
    records = collect_records(chunks_dir)
    print(f"Generated {len(records)} windows from {chunks_dir}")

    client = QdrantClient(url=QDRANT_URL)
    ensure_collection(client)
    ingest(client, records)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
