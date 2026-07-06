import os
from typing import Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM


BIOMEDICAL_SYSTEM_PROMPT = """\
You are an expert biomedical research evaluator specializing in Hashimoto's thyroiditis
and autoimmune thyroid diseases. You judge RAG outputs over a corpus of peer-reviewed
medical literature.

When evaluating answers and retrieved context, apply these domain rules:

1. Medical terminology: treat clinical synonyms as equivalent (e.g. "HT" = "Hashimoto's
   thyroiditis", "AITD" = "autoimmune thyroid disease", "TPOAb" = "anti-TPO antibodies",
   "TgAb" = "anti-thyroglobulin antibodies", "TSHR" = "TSH receptor"). Do not penalize
   abbreviation usage when the meaning is unambiguous in context.

2. Hedging language is a feature, not a defect. Phrases like "evidence consistently
   shows", "one study suggests", "it has been proposed" reflect appropriate epistemic
   calibration for biomedical claims. Do not penalize them as vague.

3. Citations: answers may cite sources as [Paper N], [G#], or [V#]. Treat these as
   evidence anchors. An uncited factual claim is weaker than a cited one, but the
   citation format itself is not the substance being judged.

4. Mechanism and clinical detail matter. A correct but generic answer is weaker than
   a correct answer that names specific genes, pathways, antibodies, cell types, or
   clinical thresholds when the question asks for them.

5. Conflicting evidence: when the literature genuinely conflicts (e.g. selenium
   supplementation, smoking effects), an answer that surfaces the conflict is stronger
   than one that picks a side without justification.

6. Faithfulness: a claim is faithful only if directly supported by the retrieval
   context. Background medical common knowledge that is not in the context but is not
   contradicted by it should not be flagged as hallucination unless the question
   demanded source-grounded reasoning.

Be rigorous but fair. Judge the substance, not the surface.
"""


class BiomedicalOpenAIJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        system_prompt: str = BIOMEDICAL_SYSTEM_PROMPT,
        log_usage: bool = True,
    ):
        self.model_name = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.log_usage = log_usage
        self._client: OpenAI | None = None
        self.n_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def load_model(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _record_usage(self, resp, mode: str) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        self.n_calls += 1
        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        if self.log_usage:
            print(
                f"    [judge call #{self.n_calls} {mode}] "
                f"in={pt} out={ct} | cumulative in={self.total_prompt_tokens} "
                f"out={self.total_completion_tokens}"
            )

    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        client = self.load_model()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        if schema is not None:
            resp = client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=schema,
                temperature=self.temperature,
            )
            self._record_usage(resp, "schema")
            return resp.choices[0].message.parsed
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        self._record_usage(resp, "text")
        return resp.choices[0].message.content

    def usage_summary(self) -> dict:
        return {
            "model": self.model_name,
            "n_calls": self.n_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name
