import os
import time

from dotenv import load_dotenv
from neo4j import GraphDatabase

from retriever.config.ret_config import NEO4J_URI, NEO4J_DATABASE, SEP, SEP2
from retriever.graph_ret.embed import embed

load_dotenv()
_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))


_FETCH_CYPHER = """
MATCH (s:Entity)-[:POSITIVE_CLAIM|NEGATIVE_CLAIM|HYPOTHETICAL_CLAIM]->(c:Claim)
                -[:CLAIM_TARGET]->(t:Entity)
OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(ev:Evidence)
WITH c, s, t,
     [e IN collect(ev.evidence_text) WHERE e IS NOT NULL AND e <> ""][..2] AS top_ev
RETURN c.claim_signature AS sig,
       s.canonical_name  AS source_name,
       c.relation_type   AS relation_type,
       t.canonical_name  AS target_name,
       c.embedding_text  AS existing_text,
       top_ev
"""

_WRITE_CYPHER = """
MATCH (c:Claim {claim_signature: $sig})
SET c.embedding_text = $text, c.embedding = $vec
"""


def _build_text(source: str, relation: str, target: str, top_ev: list[str]) -> str:
    parts = [f"{source} {relation} {target}"]
    parts.extend(top_ev)
    return ". ".join(parts)


def backfill_claim_embeddings() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=_AUTH)
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(_FETCH_CYPHER).data()
            total = len(rows)
            print(f"  [embed_claims] {total} claims found")
            print(SEP2)

            embedded = 0
            skipped  = 0
            failed   = 0
            t0 = time.time()

            for i, row in enumerate(rows, 1):
                text = _build_text(
                    row["source_name"],
                    row["relation_type"],
                    row["target_name"],
                    row["top_ev"],
                )

                if row["existing_text"] == text:
                    skipped += 1
                    print(f"  [{i}/{total}] SKIP  {row['sig'][:60]}", end="\r")
                    continue

                try:
                    vec = embed(text)
                    session.run(
                        _WRITE_CYPHER,
                        sig=row["sig"],
                        text=text,
                        vec=vec,
                    )
                    embedded += 1
                    print(f"  [{i}/{total}] EMBED {row['sig'][:60]}", end="\r")
                except Exception as e:
                    failed += 1
                    print(f"\n  [{i}/{total}] FAIL  {row['sig']}: {e}")

            dt = time.time() - t0
            print()
            print(SEP2)
            print(f"  embedded={embedded}  skipped={skipped}  failed={failed}  time={dt:.1f}s")
    finally:
        driver.close()


if __name__ == "__main__":
    print(SEP)
    print("BACKFILL: claim embeddings")
    print(SEP)
    backfill_claim_embeddings()
    print(SEP)
