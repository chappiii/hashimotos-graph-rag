from retriever.config.ret_config import GENERIC_ENTITIES, SEP, SEP2
from retriever.graph_ret.query_decomposer import decompose
from retriever.graph_ret.entity_matcher import load_registry, match_entities
from retriever.graph_ret.graph_retriever import get_claims
from retriever.graph_ret.ranker import rank_by_quality, rank_by_hybrid
from retriever.vector_ret.retriever import retrieve

QUERIES = [
    "What is the effect of Vitamin D supplementation on TSH levels in Hashimoto's Thyroiditis patients?",
    # "How does selenium affect thyroid antibodies in HT?",
    # "Is there a link between gluten and autoimmune thyroid disease?",
    # "What is the relationship between TSH and thyroid peroxidase antibodies?",
    # "What immune cells are involved in the pathogenesis of Hashimoto's Thyroiditis?",
]


def run_graph_ret(query: str, registry: dict) -> list[dict]:
    spans   = decompose(query)
    matched = match_entities(spans, registry)
    claims  = get_claims(matched)
    specific = [e for e in matched if e not in GENERIC_ENTITIES]
    quality_top = rank_by_quality(claims, specific_entities=specific)
    return rank_by_hybrid(quality_top, query, specific_entities=specific)


def print_graph_results(claims: list[dict]) -> None:
    print(f"\n  GRAPH RETRIEVER  ({len(claims)} claims)")
    print(SEP2)
    for i, c in enumerate(claims, 1):
        print(f"  [{i}] {c['source_name']} --[{c['relation_type']}]--> {c['target_name']}")
        print(f"       score={c['_hybrid_score']:.3f}  certainty={c['certainty_max']}  papers={c['paper_count']}")
        for ev in c.get("evidence_list", []):
            print(f"       • [{ev.get('section_type','?')} | {ev.get('study_design','?')} | {ev.get('paper_year','?')}]")
            print(f"         {ev['evidence_text'][:140]}")


def print_vector_results(chunks: list[dict]) -> None:
    print(f"\n  VECTOR RETRIEVER  ({len(chunks)} chunks)")
    print(SEP2)
    for i, h in enumerate(chunks, 1):
        print(f"  [{i}] score={h['score']:.4f}  paper={h['paper_id']}  [{h['section_type']}]  {h['section_name']}  (win {h['window_index']+1}/{h['window_count']})")
        print(f"       {h['text'][:200]}")


def main() -> None:
    registry = load_registry()

    query = QUERIES[0] 

    print(SEP)
    print(f"QUERY: {query}")
    print(SEP)

    graph_results  = run_graph_ret(query, registry)
    vector_results = retrieve(query)

    print_graph_results(graph_results)
    print_vector_results(vector_results)
    print(SEP)


if __name__ == "__main__":
    main()
