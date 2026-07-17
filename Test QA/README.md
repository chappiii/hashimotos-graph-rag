# Test QA

The exact multiple-choice question sets used to evaluate this system. They are
published here so the reported accuracy numbers can be checked against the
questions they were measured on.

Each file is a Hashimoto's thyroiditis relevant subset of a public medical QA
benchmark, retrieved from Hugging Face. All four upstream datasets carry
permissive licenses that allow redistribution with attribution.

## Contents

| File | Benchmark | Items | Options | Question style |
|---|---|---:|---|---|
| [`ht_medmcqa_relevant-final.json`](ht_medmcqa_relevant-final.json) | MedMCQA | 86 | A-D | Single-fact recall (AIIMS / NEET PG entrance exam) |
| [`ht_medqa-final.json`](ht_medqa-final.json) | MedQA | 28 | A-D | USMLE Step 1/2/3 clinical vignettes |
| [`medbullets-final.json`](medbullets-final.json) | Medbullets | 7 | A-E | USMLE Step 2 clinical vignettes |
| [`medxpertqa-final.json`](medxpertqa-final.json) | MedXpertQA | 25 | A-J | Specialist-level reasoning |

146 questions total. Every file is a flat JSON array of objects.

## Sources and licensing

| Benchmark | Hugging Face dataset | License |
|---|---|---|
| MedMCQA | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Apache-2.0 |
| MedQA | [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | CC-BY-4.0 |
| Medbullets | [`JesseLiu/medbulltes5op`](https://huggingface.co/datasets/JesseLiu/medbulltes5op) | Apache-2.0 |
| MedXpertQA | [`TsinghuaC3I/MedXpertQA`](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) (Text subset) | MIT |

Licenses are as declared on each dataset card at time of retrieval. Note that
three of the four Hugging Face repositories are community re-uploads rather than
the original authors' own distributions; `TsinghuaC3I/MedXpertQA` is the
exception. Attribution below therefore credits the original papers, which is the
citation that matters in either case.

Question text, options, answer keys and explanations are the work of the
respective upstream authors and remain under their licenses. Nothing in these
four files was written by this project. The only contribution here is the
selection of which items are relevant to Hashimoto's thyroiditis.

## Schemas

Field names are inherited from upstream and left untouched, so these files stay
diff-able against the source datasets.

### `ht_medmcqa_relevant-final.json`

| Field | Meaning |
|---|---|
| `id` | Upstream MedMCQA UUID |
| `question` | Question stem |
| `opa`, `opb`, `opc`, `opd` | Options A through D |
| `cop` | Correct option, **0-indexed**: 0 = `opa`, 1 = `opb`, 2 = `opc`, 3 = `opd` |
| `exp` | Upstream explanation of the answer |
| `subject_name`, `topic_name` | Upstream taxonomy (e.g. Medicine / Endocrinology) |
| `choice_type` | Upstream `single` or `multi` flag, carried through as-is |

### `ht_medqa-final.json`

| Field | Meaning |
|---|---|
| `question` | Clinical vignette |
| `options` | Object mapping `"A"`-`"D"` to option text |
| `answer` | Correct option text |
| `answer_idx` | Correct option letter |
| `meta_info` | Upstream USMLE step label (e.g. `step2&3`) |

This is the only set without an upstream `id` field; the source dataset does not
provide one.

### `medbullets-final.json`

| Field | Meaning |
|---|---|
| `id` | Upstream Medbullets item id |
| `split` | Upstream split label |
| `question` | Clinical vignette |
| `options` | Object mapping `"A"`-`"E"` to option text |
| `answer` | Correct option text |
| `answer_idx` | Correct option letter |
| `explanation` | Upstream explanation of the answer |
| `link` | Source URL on medbullets.com |

### `medxpertqa-final.json`

| Field | Meaning |
|---|---|
| `id` | Upstream MedXpertQA id (e.g. `Text-86`) |
| `split` | Upstream split label |
| `question` | Clinical vignette |
| `options` | Object mapping `"A"`-`"J"` to option text |
| `answer` | Correct option text |
| `answer_idx` | Correct option letter |
| `medical_task` | Upstream label: Diagnosis, Treatment, or Basic Science |
| `body_system` | Upstream label: Endocrine, Nervous, or Skeletal |
| `question_type` | Upstream label: Reasoning or Understanding |

## Citation

If you use these subsets, cite the original benchmarks.

```bibtex
@InProceedings{pmlr-v174-pal22a,
  title     = {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author    = {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  pages     = {248--260},
  year      = {2022},
  volume    = {174},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v174/pal22a.html}
}

@article{jin2020disease,
  title   = {What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author  = {Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal = {arXiv preprint arXiv:2009.13081},
  year    = {2020}
}

@article{chen2024benchmarking,
  title   = {Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions},
  author  = {Chen, Hanjie and Fang, Zhouxiang and Singla, Yash and Dredze, Mark},
  journal = {arXiv preprint arXiv:2402.18060},
  year    = {2024}
}

@article{zuo2025medxpertqa,
  title   = {MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding},
  author  = {Zuo, Yuxin and Qu, Shang and Li, Yifei and Chen, Zhangren and Zhu, Xuekai and Hua, Ermo and Zhang, Kaiyan and Ding, Ning and Zhou, Bowen},
  journal = {arXiv preprint arXiv:2501.18362},
  year    = {2025}
}
```
