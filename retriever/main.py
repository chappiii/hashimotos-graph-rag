from retriever.config.ret_config import GENERIC_ENTITIES, MAX_PATH_HOPS, SEP, SEP2
from retriever.graph_ret.query_decomposer import decompose
from retriever.graph_ret.entity_matcher import load_registry, match_entities
from retriever.graph_ret.graph_retriever import get_paths
from retriever.graph_ret.path_expander import expand_paths
from retriever.graph_ret.ranker import rank_by_quality, rank_by_hybrid
from retriever.vector_ret.retriever import retrieve
from generator.generator import generate

QUERIES = [
    "What is the effect of Vitamin D supplementation on TSH levels in Hashimoto's Thyroiditis patients?",
    # "What is the relationship between TSH and thyroid peroxidase antibodies?"
    # "How does selenium affect thyroid antibodies in HT?",
    # "Is there a link between gluten and autoimmune thyroid disease?",
    # "What is the relationship between TSH and thyroid peroxidase antibodies?",
    # "What immune cells are involved in the pathogenesis of Hashimoto's Thyroiditis?",
]


def _merge_paths(*lists: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for lst in lists:
        for p in lst:
            merged.setdefault(p["path_signature"], p)
    return list(merged.values())


def run_graph_ret(query: str, registry: dict) -> list[dict]:
    spans   = decompose(query)
    matched = match_entities(spans, registry)

    direct_paths = get_paths(matched)
    chain_paths  = expand_paths(matched) if MAX_PATH_HOPS >= 2 else []
    print(f"  [graph] direct(1-hop)={len(direct_paths)}  chains(2-hop)={len(chain_paths)}")

    all_paths   = _merge_paths(direct_paths, chain_paths)
    specific    = [e for e in matched if e not in GENERIC_ENTITIES]
    quality_top = rank_by_quality(all_paths, specific_entities=specific)
    return rank_by_hybrid(quality_top, query)


def _chain_str(path: dict) -> str:
    parts = [path["entities"][0]]
    for i, c in enumerate(path["claims"], 1):
        parts.append(f"--[{c['relation_type']}]-->")
        parts.append(path["entities"][i])
    return " ".join(parts)


def print_graph_results(paths: list[dict]) -> None:
    direct = sum(1 for p in paths if p["length"] == 1)
    chains = sum(1 for p in paths if p["length"] == 2)
    print(f"\n  GRAPH RETRIEVER  ({len(paths)} paths: {direct} direct, {chains} chains)")
    print(SEP2)
    for i, p in enumerate(paths, 1):
        kind = "DIRECT" if p["length"] == 1 else "CHAIN "
        print(f"  [{i}] {kind} {_chain_str(p)}")
        print(f"       score={p['_hybrid_score']:.3f}  quality={p['_quality_score']:.3f}  anchors={p['anchor_entities']}")
        for j, c in enumerate(p["claims"], 1):
            tag = "claim" if p["length"] == 1 else f"hop {j}"
            print(f"       [{tag}] certainty={c['certainty_max']}  papers={c['paper_count']}")
            for ev in c.get("evidence_list", []):
                print(f"         . [{ev.get('section_type','?')} | {ev.get('study_design','?')} | {ev.get('paper_year','?')}]")
                print(f"           {ev['evidence_text'][:140]}")


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

    graph_paths    = run_graph_ret(query, registry)
    vector_results = retrieve(query)

    print_graph_results(graph_paths)
    print_vector_results(vector_results)
    print(SEP)

    print("\n  GENERATOR")
    print(SEP2)
    answer = generate(query, graph_paths, vector_results)
    print(answer)
    print(SEP)


if __name__ == "__main__":
    main()
