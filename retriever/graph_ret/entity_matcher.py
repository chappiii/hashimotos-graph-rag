import json
import math
import re

from rapidfuzz import fuzz

from retriever.config.ret_config import (
    ENTITY_MATCH_THRESHOLD,
    ENTITY_TOP_K,
    MIN_MATCH_COVERAGE,
    ENTITY_REGISTRY_PATH,
)


def load_registry() -> dict:
    with open(ENTITY_REGISTRY_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("entities", raw)


def match_entities(
    spans: list[str],
    registry: dict,
    threshold: int = ENTITY_MATCH_THRESHOLD,
    top_k: int = ENTITY_TOP_K,
) -> list[str]:
    best_scores: dict[str, float] = {}

    for canonical, data in registry.items():
        all_names = [canonical] + (data.get("aliases") or [])

        best = 0.0
        for span in spans:
            span_tokens = len(span.split())
            for name in all_names:
                name_tokens = len(re.findall(r"[A-Za-z0-9']+", name))
                if span_tokens < math.ceil(name_tokens * MIN_MATCH_COVERAGE):
                    continue
                # short names: use ratio to prevent substring false matches
                if len(name) < 5:
                    s = fuzz.ratio(span.lower(), name.lower())
                else:
                    s = fuzz.WRatio(span.lower(), name.lower())
                if s > best:
                    best = s
                if best == 100:
                    break
            if best == 100:
                break

        if best >= threshold:
            best_scores[canonical] = best

    ranked = sorted(best_scores.items(), key=lambda x: -x[1])
    return [name for name, _ in ranked[:top_k]]
