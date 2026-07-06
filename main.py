"""
Graph RAG over Hashimoto's Thyroiditis literature - pipeline entrypoint.

Two subcommands:

  build   Run the ingestion pipeline against the LOCAL databases:
          registries -> Neo4j (graph) -> Qdrant (vectors).

  query   Run the live flow for one question:
          decompose -> entity match -> graph retrieve (1-hop + 2-hop) ->
          rank -> vector retrieve -> generate a cited answer.

The upstream extraction stages (metadata, section chunking, entity/relation
extraction) are long Gemini batch jobs whose outputs are already committed to
the repo. They are opt-in via `build --with-extraction`.

Every stage is launched as its own process, exactly how it runs standalone, so
skip logic, argv, and per-stage configs behave identically to running them by
hand. Run this from the repo root: the entity registry path is resolved
relative to the working directory.

Examples:
  python main.py build
  python main.py build --with-extraction
  python main.py query "What is the effect of selenium on thyroid antibodies in HT?"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEP = "=" * 72

# Each stage runs in its own process. Two launch shapes:
#   "module" -> python -m <module>            (cwd = repo root)
#   "script" -> python <script> (+ "cwd")     (for the one stage that expects
#                                              to run from inside its own dir)
EXTRACTION_STAGES = [
    ("metadata extraction",        {"script": "main.py", "cwd": ROOT / "extract_metadata"}),
    ("pdf section chunking",       {"module": "pdf_section_chunker.main"}),
    ("entity/relation extraction", {"module": "extract_entity_relation.main", "args": ["--model", "gemini"]}),
]

# Order matters: registries feed Neo4j; Neo4j schema/nodes before relations.
INGESTION_STAGES = [
    ("entity registry",     {"module": "kg_ingestion.pre_ingestion.entity_registry"}),
    ("claim registry",      {"module": "kg_ingestion.pre_ingestion.claim_registry"}),
    ("neo4j: schema",       {"module": "kg_ingestion.neo4j.schema"}),
    ("neo4j: papers",       {"module": "kg_ingestion.neo4j.ingest_papers"}),
    ("neo4j: entities",     {"module": "kg_ingestion.neo4j.ingest_entities"}),
    ("neo4j: claims",       {"module": "kg_ingestion.neo4j.ingest_claims"}),
    ("neo4j: evidence",     {"module": "kg_ingestion.neo4j.ingest_evidence"}),
    ("qdrant: collections", {"module": "kg_ingestion.qdrant.collections"}),
    ("qdrant: papers",      {"module": "kg_ingestion.qdrant.ingest_papers"}),
    ("qdrant: chunks",      {"module": "kg_ingestion.qdrant.ingest_chunks"}),
    ("qdrant: evidence",    {"module": "kg_ingestion.qdrant.ingest_evidence"}),
]


def run_stage(name: str, spec: dict) -> None:
    if "module" in spec:
        cmd = [sys.executable, "-m", spec["module"], *spec.get("args", [])]
        cwd = ROOT
    else:
        cmd = [sys.executable, spec["script"], *spec.get("args", [])]
        cwd = spec["cwd"]

    print(f"\n{SEP}\n>>> {name}\n    {' '.join(str(c) for c in cmd)}\n{SEP}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"\nStage '{name}' failed (exit {result.returncode}). Aborting.")
    print(f"    [{name}] done in {time.time() - start:.1f}s")


def build(with_extraction: bool) -> None:
    stages = (EXTRACTION_STAGES if with_extraction else []) + INGESTION_STAGES
    label = "full (extraction + ingestion)" if with_extraction else "ingestion only"
    print(f"BUILD - {label} - {len(stages)} stages - local Neo4j + Qdrant")

    run_start = time.time()
    for name, spec in stages:
        run_stage(name, spec)

    print(f"\n{SEP}\nBUILD COMPLETE - {len(stages)} stages in "
          f"{time.time() - run_start:.1f}s\n{SEP}")


def query(question: str) -> None:
    # Imported here so `build` (subprocess-based) does not need the retrieval deps.
    from retriever.config.ret_config import SEP2
    from retriever.graph_ret.entity_matcher import load_registry
    from retriever.vector_ret.retriever import retrieve
    from retriever.main import run_graph_ret, print_graph_results, print_vector_results
    from generator.generator import generate

    registry = load_registry()

    print(SEP)
    print(f"QUERY: {question}")
    print(SEP)

    graph_paths = run_graph_ret(question, registry)
    vector_results = retrieve(question)

    print_graph_results(graph_paths)
    print_vector_results(vector_results)
    print(SEP)

    print("\n  GENERATOR")
    print(SEP2)
    answer = generate(question, graph_paths, vector_results)
    print(answer)
    print(SEP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph RAG pipeline for Hashimoto's Thyroiditis (local DBs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Run the ingestion pipeline against local Neo4j + Qdrant")
    b.add_argument(
        "--with-extraction",
        action="store_true",
        help="Also run the upstream Gemini extraction stages (metadata, sections, entities). Long; needs GEMINI_API_KEY.",
    )

    q = sub.add_parser("query", help="Ask one question through the live retrieve + generate flow")
    q.add_argument("question", help="The natural-language question")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args.with_extraction)
    elif args.command == "query":
        query(args.question)


if __name__ == "__main__":
    main()
