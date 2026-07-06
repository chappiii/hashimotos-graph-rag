"""Run DeepEval RAG metrics on all 3 systems (hybrid, vector, vanilla).

Reads:
  - deval/data/answers.json                          (ground truth)
  - deval/data/{hybrid,vector,vanilla}_results.json  (candidate outputs)

Writes:
  - deval/eval_results/{system}_deepeval.json   (per-question + summary)

Metrics:
  - AnswerRelevancy           (all 3 systems)
  - Faithfulness              (hybrid, vector only -- needs retrieval_context)
  - ContextualPrecision       (hybrid, vector only)
  - ContextualRecall          (hybrid, vector only)
  - ContextualRelevancy       (hybrid, vector only)
  - RAGAS                     (computed as mean of the 4 above + AnswerRelevancy)
"""

import json
import traceback
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from deval.judge import BiomedicalOpenAIJudge


HERE = Path(__file__).parent
DATA_DIR = HERE / "ablation"
GT_PATH = DATA_DIR / "answers.json"
# OUTPUT_DIR = HERE / "eval_results2"
OUTPUT_DIR = HERE / "ab_results"

# ablation
SYSTEMS = {
    "no_cot":  DATA_DIR / "no_cot_results.json",
    # "no_graph":  DATA_DIR / "no_graph_results.json",
    # "no_vector": DATA_DIR / "no_vector_results.json",
}

# SYSTEMS = {
#     "hybrid":  DATA_DIR / "hybrid_results.json",
#     "vector":  DATA_DIR / "vector_results.json",
#     "vanilla": DATA_DIR / "vanilla_results.json",
# }

THRESHOLD = 0.7


def vector_chunks_to_context(chunks: list[dict]) -> list[str]:
    return [
        f"[Paper {c.get('paper_id','?')} | {c.get('section_name','?')}]\n{c.get('text','')}"
        for c in chunks
    ]


def graph_paths_to_context(paths: list[dict]) -> list[str]:
    out = []
    for p in paths:
        for claim in p.get("claims", []):
            triple = (
                f"({claim.get('source_name','?')}) "
                f"-[{claim.get('relation_type','?')}]-> "
                f"({claim.get('target_name','?')})"
            )
            for ev in claim.get("evidence_list", []):
                out.append(
                    f"{triple}\n"
                    f"Evidence (Paper {ev.get('paper_id','?')}, "
                    f"{ev.get('paper_year','?')}, "
                    f"{ev.get('section_type','?')}): "
                    f"{ev.get('evidence_text','')}"
                )
    return out


def build_retrieval_context(system: str, item: dict) -> list[str]:
    if system == "vanilla":
        return []
    ctx = item.get("context", {}) or {}
    vector = vector_chunks_to_context(ctx.get("vector_chunks", []))
    graph = graph_paths_to_context(ctx.get("graph_paths", []))
    return vector + graph


def metrics_for(system: str, judge) -> list:
    answer_rel = AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, async_mode=False)
    if system == "vanilla":
        return [answer_rel]
    return [
        answer_rel,
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, async_mode=False),
        ContextualRelevancyMetric(threshold=THRESHOLD, model=judge, async_mode=False),
    ]


def ragas_average(scores: dict) -> float | None:
    keys = [
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
    ]
    vals = [scores[k] for k in keys if scores.get(k) is not None]
    if len(vals) != 4:
        return None
    return sum(vals) / 4


def summarize(per_question: list[dict]) -> dict[str, float | None]:
    metric_names = set()
    for q in per_question:
        metric_names.update(q["scores"].keys())
    summary = {}
    for m in metric_names:
        vals = [q["scores"][m] for q in per_question if q["scores"].get(m) is not None]
        summary[m] = (sum(vals) / len(vals)) if vals else None
    return summary


def run_system(system: str, path: Path, gt_by_id: dict, judge) -> dict:
    cand_items = json.loads(path.read_text(encoding="utf-8"))
    per_question = []

    for i, item in enumerate(cand_items, 1):
        qid = item["id"]
        gt = gt_by_id.get(qid)
        if not gt or not item.get("answer"):
            print(f"  [{i:>2}/{len(cand_items)}] {qid}: missing GT or answer, skipping")
            continue

        retrieval_context = build_retrieval_context(system, item)
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=item["answer"],
            expected_output=gt["answer"],
            retrieval_context=retrieval_context if retrieval_context else None,
        )

        metrics = metrics_for(system, judge)
        scores: dict[str, float | None] = {}
        details = []

        for metric in metrics:
            name = metric.__class__.__name__
            try:
                metric.measure(test_case)
                scores[name] = float(metric.score) if metric.score is not None else None
                details.append({
                    "name": name,
                    "score": scores[name],
                    "threshold": metric.threshold,
                    "success": bool(metric.success) if metric.success is not None else None,
                    "reason": metric.reason,
                })
            except Exception as e:
                scores[name] = None
                details.append({
                    "name": name,
                    "score": None,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(limit=2),
                })

        scores["RAGAS"] = ragas_average(scores)

        score_str = "  ".join(
            f"{k.replace('Metric','')}={v:.3f}" if v is not None else f"{k.replace('Metric','')}=NA"
            for k, v in scores.items()
        )
        print(f"  [{i:>2}/{len(cand_items)}] {qid:<3} [{item.get('category','?'):<13}] {score_str}")

        per_question.append({
            "id": qid,
            "category": item.get("category", gt.get("category")),
            "scores": scores,
            "details": details,
        })

    return {
        "system": system,
        "judge": judge.get_model_name(),
        "n_questions": len(per_question),
        "threshold": THRESHOLD,
        "summary": summarize(per_question),
        "per_question": per_question,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    judge = BiomedicalOpenAIJudge()
    gt_items = json.loads(GT_PATH.read_text(encoding="utf-8"))["answers"]
    gt_by_id = {g["id"]: g for g in gt_items}

    print(f"Judge: {judge.get_model_name()}")
    print(f"Questions in GT: {len(gt_items)}")

    overall = {}
    usage_per_system = {}
    for system, path in SYSTEMS.items():
        print()
        print("=" * 72)
        print(f"SYSTEM: {system}   ({path.name})")
        print("=" * 72)
        before = judge.usage_summary()
        result = run_system(system, path, gt_by_id, judge)
        after = judge.usage_summary()
        result["judge_usage"] = {
            "n_calls": after["n_calls"] - before["n_calls"],
            "prompt_tokens": after["prompt_tokens"] - before["prompt_tokens"],
            "completion_tokens": after["completion_tokens"] - before["completion_tokens"],
        }
        usage_per_system[system] = result["judge_usage"]
        out_path = OUTPUT_DIR / f"{system}_deepeval.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  saved -> {out_path}")
        overall[system] = result["summary"]

    print()
    print("=" * 72)
    print("SUMMARY (mean scores)")
    print("=" * 72)
    metric_order = [
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
        "ContextualRelevancyMetric",
        "RAGAS",
    ]
    header = f"{'system':<10} " + " ".join(f"{m.replace('Metric',''):>12}" for m in metric_order)
    print(header)
    print("-" * len(header))
    for system, summary in overall.items():
        row = f"{system:<10} " + " ".join(
            f"{summary.get(m):>12.3f}" if summary.get(m) is not None else f"{'NA':>12}"
            for m in metric_order
        )
        print(row)

    print()
    print("=" * 72)
    print(f"JUDGE TOKEN USAGE  ({judge.get_model_name()})")
    print("=" * 72)
    print(f"{'system':<10} {'calls':>8} {'prompt_tok':>12} {'compl_tok':>12} {'total_tok':>12}")
    print("-" * 58)
    for system, u in usage_per_system.items():
        total = u["prompt_tokens"] + u["completion_tokens"]
        print(f"{system:<10} {u['n_calls']:>8} {u['prompt_tokens']:>12} {u['completion_tokens']:>12} {total:>12}")
    grand = judge.usage_summary()
    print("-" * 58)
    print(f"{'TOTAL':<10} {grand['n_calls']:>8} {grand['prompt_tokens']:>12} "
          f"{grand['completion_tokens']:>12} {grand['total_tokens']:>12}")


if __name__ == "__main__":
    main()
