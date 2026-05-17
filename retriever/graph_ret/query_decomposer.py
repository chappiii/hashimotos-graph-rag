import re

from retriever.config.ret_config import NGRAM_MAX_N, NGRAM_MIN_UNIGRAM, STOPWORDS


def decompose(query: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", query)
    seen: set[str] = set()
    spans: list[str] = []

    for n in range(1, min(NGRAM_MAX_N + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]

            if n == 1:
                tok = window[0]
                if len(tok) < NGRAM_MIN_UNIGRAM:
                    continue
                if tok.lower() in STOPWORDS:
                    continue
            else:
                if all(t.lower() in STOPWORDS for t in window):
                    continue

            span = " ".join(window)
            lower = span.lower()
            if lower not in seen:
                seen.add(lower)
                spans.append(span)

    return spans
