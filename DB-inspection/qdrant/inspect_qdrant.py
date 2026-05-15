"""
Inspection script for the 3 Qdrant collections: papers, chunks, evidence.

Avoids bulk scroll on chunks/evidence (Qdrant OutputTooSmall panic with large text payloads).
Uses filter-based count() on indexed fields for all aggregate stats.

Run: uv run DB-inspection/qdrant/inspect_qdrant.py
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from kg_ingestion.config.kg_config import QDRANT_URL
from kg_ingestion.qdrant.embed import embed

COLLECTIONS  = ["papers", "chunks", "evidence"]
PAPER_IDS    = range(1, 116)
SECTION_TYPES = ["ABSTRACT", "INTRODUCTION", "METHODS", "RESULTS", "DISCUSSION", "CONCLUSION", "OTHER"]
POLARITIES   = ["positive", "negative", "mixed", "uncertain", "hypothetical"]
CERTAINTIES  = ["high", "moderate", "low"]


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def _count(client: QdrantClient, collection: str, field: str, value) -> int:
    return client.count(
        collection_name=collection,
        count_filter=Filter(must=[FieldCondition(key=field, match=MatchValue(value=value))]),
        exact=True,
    ).count


def _sample(client: QdrantClient, collection: str, n: int = 3) -> list:
    """Fetch n points one at a time to avoid the bulk-scroll panic."""
    points = []
    offset = None
    for _ in range(n):
        batch, offset = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not batch:
            break
        points.append(batch[0])
    return points


# ---------------------------------------------------------------------------
# 1. overview
# ---------------------------------------------------------------------------

def overview(client: QdrantClient) -> None:
    section("1. OVERVIEW")

    for name in COLLECTIONS:
        try:
            info  = client.get_collection(name)
            count = client.count(collection_name=name).count
        except Exception as e:
            print(f"\n{name}: NOT FOUND ({e})")
            continue

        try:
            vp       = info.config.params.vectors
            dim      = getattr(vp, "size", "?")
            distance = getattr(vp, "distance", "?")
        except Exception:
            dim = distance = "?"

        indexed      = list((info.payload_schema or {}).keys())
        sample_pt    = _sample(client, name, n=1)
        payload_keys = list((sample_pt[0].payload or {}).keys()) if sample_pt else []

        print(f"\n{name}:")
        print(f"  points:          {count:,}")
        print(f"  vector dim:      {dim}  distance: {distance}")
        print(f"  indexed fields:  {indexed}")
        print(f"  payload keys:    {payload_keys}")


# ---------------------------------------------------------------------------
# 2. per-paper coverage
# ---------------------------------------------------------------------------

def per_paper_coverage(client: QdrantClient) -> None:
    section("2. PER-PAPER COVERAGE")

    for collection in ["chunks", "evidence"]:
        counts = {pid: _count(client, collection, "paper_id", pid) for pid in PAPER_IDS}
        covered = {pid: n for pid, n in counts.items() if n > 0}
        missing = sorted(pid for pid, n in counts.items() if n == 0)
        total   = sum(covered.values())

        print(f"\n{collection}:")
        print(f"  papers covered:  {len(covered)} / 115")
        print(f"  total points:    {total:,}")
        if covered:
            print(f"  avg per paper:   {total / len(covered):.1f}")
            print(f"  min / max:       {min(covered.values())} / {max(covered.values())}")
        if missing:
            print(f"  papers missing:  {missing}")


# ---------------------------------------------------------------------------
# 3. section_type distribution (chunks)
# ---------------------------------------------------------------------------

def section_type_distribution(client: QdrantClient) -> None:
    section("3. CHUNK SECTION_TYPE DISTRIBUTION")

    total = client.count(collection_name="chunks").count
    counts = {stype: _count(client, "chunks", "section_type", stype) for stype in SECTION_TYPES}

    for stype, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * int(n / total * 40) if total else ""
        print(f"  {stype:<15} {n:>5}  {bar}")


# ---------------------------------------------------------------------------
# 4. evidence polarity & certainty breakdown
# ---------------------------------------------------------------------------

def evidence_breakdown(client: QdrantClient) -> None:
    section("4. EVIDENCE POLARITY & CERTAINTY BREAKDOWN")

    print("\nPolarity:")
    for pol in POLARITIES:
        n = _count(client, "evidence", "claim_polarity", pol)
        print(f"  {pol:<15} {n:>5}")

    print("\nCertainty:")
    for cert in CERTAINTIES:
        n = _count(client, "evidence", "claim_certainty", cert)
        print(f"  {cert:<15} {n:>5}")


# ---------------------------------------------------------------------------
# 5. sample records
# ---------------------------------------------------------------------------

def sample_records(client: QdrantClient) -> None:
    section("5. SAMPLE RECORDS")

    print("\n--- papers (3 samples) ---")
    for p in _sample(client, "papers", n=3):
        pl = p.payload or {}
        print(f"  [{pl.get('paper_id')}] {str(pl.get('title', ''))[:80]}")
        print(f"       study_design={pl.get('study_design')}  year={pl.get('year')}")
        print(f"       purpose: {str(pl.get('purpose_of_work', ''))[:120]}...")

    print("\n--- chunks (3 samples) ---")
    for p in _sample(client, "chunks", n=3):
        pl = p.payload or {}
        print(f"  paper={pl.get('paper_id')}  {pl.get('section_type')}  win {pl.get('window_index')}/{(pl.get('window_count') or 1) - 1}  [{pl.get('chunk_filename')}]")
        print(f"  text: {str(pl.get('text', ''))[:160]}...")

    print("\n--- evidence (3 samples) ---")
    for p in _sample(client, "evidence", n=3):
        pl = p.payload or {}
        print(f"  paper={pl.get('paper_id')}  {pl.get('claim_polarity')}/{pl.get('claim_certainty')}")
        print(f"  {pl.get('source_entity')} --[{pl.get('relation_type')}]--> {pl.get('target_entity')}")
        print(f"  \"{str(pl.get('evidence_text', ''))[:160]}\"")


# ---------------------------------------------------------------------------
# 6. semantic similarity test
# ---------------------------------------------------------------------------

def semantic_test(client: QdrantClient) -> None:
    section("6. SEMANTIC SIMILARITY TEST")

    query = "Vitamin D supplementation effect on anti-TPO antibodies in Hashimoto's thyroiditis"
    print(f"Query: {query}\n")

    vec = embed(query, task_type="RETRIEVAL_QUERY")

    for name in COLLECTIONS:
        print(f"--- top 3 from {name} ---")
        results = client.query_points(
            collection_name=name,
            query=vec,
            limit=3,
            with_payload=True,
        ).points
        for r in results:
            pl = r.payload or {}
            if name == "papers":
                print(f"  {r.score:.3f}  [{pl.get('paper_id')}] {str(pl.get('title', ''))[:80]}")
            elif name == "chunks":
                print(f"  {r.score:.3f}  paper={pl.get('paper_id')}  {pl.get('section_type')}")
                print(f"         {str(pl.get('text', ''))[:140]}...")
            else:
                print(f"  {r.score:.3f}  paper={pl.get('paper_id')}  {pl.get('source_entity')} -> {pl.get('target_entity')}")
                print(f"         \"{str(pl.get('evidence_text', ''))[:140]}\"")


# ---------------------------------------------------------------------------
# 7. quality checks
# ---------------------------------------------------------------------------

def quality_checks(client: QdrantClient) -> None:
    section("7. QUALITY CHECKS")

    print("payload completeness (papers — scrollable, small payloads):")
    fields = ["title", "doi", "year", "study_design", "countries", "keywords", "purpose_of_work"]
    field_counts = {f: 0 for f in fields}
    total = 0
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name="papers",
            limit=50,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for p in batch:
            total += 1
            pl = p.payload or {}
            for f in fields:
                v = pl.get(f)
                if v not in (None, "", [], {}):
                    field_counts[f] += 1
        if offset is None:
            break

    for f in fields:
        n = field_counts[f]
        pct = f"{n / total * 100:.0f}%" if total else "—"
        print(f"  {f:<20} {n}/{total}  ({pct})")

    print("\npoint counts match expected:")
    print(f"  papers:   {client.count('papers').count} / 115 expected")
    print(f"  chunks:   {client.count('chunks').count} / 1765 expected")
    print(f"  evidence: {client.count('evidence').count} / 5524 expected")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    print(f"Connected to {QDRANT_URL}")

    overview(client)
    per_paper_coverage(client)
    section_type_distribution(client)
    evidence_breakdown(client)
    sample_records(client)
    semantic_test(client)
    quality_checks(client)


if __name__ == "__main__":
    main()
