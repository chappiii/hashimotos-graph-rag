extract_relation_prompt = """
You are a top-tier biomedical information extraction algorithm specialized in identifying relationships between clinical entities for knowledge graph construction.

You are given (a) a list of pre-extracted entities and (b) the source text. Your task is to extract relationships between those entities that are EXPLICITLY CLAIMED in the text.

For each relation, return:
- `relation_type` — must EXACTLY match one of the snake_case names from the table below
- `source_entity` — the source entity (must be in the pre-extracted list)
- `target_entity` — the target entity (must be in the pre-extracted list)
- `key_properties` — properties relevant to the relation (use `null` for unknown values; do NOT omit fields)
- `evidence` — a single verbatim sentence from the text supporting the relation
- `claim_polarity` — whether the relation is affirmed or denied: `positive` | `negative` | `mixed` | `uncertain` | `hypothetical`
- `claim_certainty` — strength of the claim: `high` | `moderate` | `low`

## Allowed Relations (32 types)

Direction matters. The `direction` column shows allowed `source_type → target_type` patterns. Do NOT reverse them.

| relation_type                   | description                                                                | direction (source → target)                                                                                                                          | example                          | key_properties                                       |
| ------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------- |
| causes_or_increases_risk_of     | One entity causes or increases the likelihood of another.                  | Diseases / Genes / Genetic Variant / Comorbidities / Lifestyle Factor / Environmental Factor / Hormones / Treatments / Medical Event / Mechanisms → Diseases / Cancer / Symptoms | HT → Hypothyroidism              | risk_strength, evidence_type                         |
| associated_with                 | Statistically or clinically meaningful link without claimed causality.     | Any clinical entity → any clinical entity                                                                                                            | Vitamin D deficiency → HT        | association_strength, association_type               |
| negatively_correlated_with      | Inverse correlation between two measurable variables.                      | Hormones / Biomarkers / Lab Findings → Hormones / Biomarkers / Lab Findings                                                                          | 25(OH)D → anti-TPO               | correlation_coefficient                              |
| positively_correlated_with      | Positive correlation between two measurable variables.                     | Hormones / Biomarkers / Lab Findings → Hormones / Biomarkers / Lab Findings                                                                          | TgAb → Triglycerides             | correlation_coefficient                              |
| treated_with                    | A treatment used to manage a condition.                                    | Diseases / Cancer → Treatments                                                                                                                       | HT → Levothyroxine               | treatment_type, effectiveness, is_first_line         |
| improves_or_reduces             | Intervention reduces or improves a measurable parameter.                   | Treatments / Lifestyle / Environmental → Hormones / Biomarkers / Symptoms / Lab Findings                                                             | Cholecalciferol → anti-TPO       | direction_of_effect, percent_degree_of_change        |
| inhibited_by                    | Mechanism or process suppressed by an agent.                               | Mechanisms / Diseases → Treatments / Hormones / Mechanisms                                                                                           | B cell proliferation → 1,25(OH)₂D | inhibitor, target_process                            |
| induces                         | Biological mechanism triggers another process or feature.                  | Hormones / Mechanisms → Mechanisms / Pathological Features / Symptoms                                                                                | IFN-γ → HLA-DR expression        | pathway, cell_type                                   |
| diagnosed_by                    | Diagnostic method, marker, or criterion used to identify a condition.      | Diseases / Cancer → Diagnostic Methods / Hormones / Biomarkers / Lab Findings                                                                        | HT → Elevated anti-TPO           | modality_type, diagnostic_criteria                   |
| is_a_serologic_marker_of        | Biomarker or antibody used to identify a disease.                          | Hormones / Biomarkers / Antibodies → Diseases / Cancer                                                                                               | Anti-TPO → HT                    | test_type, diagnostic_role                           |
| detectable_by                   | Finding or feature detectable using a method/modality.                     | Pathological Features / Symptoms → Diagnostic Methods                                                                                                | Thyroid nodule → Ultrasound      | modality                                             |
| has_feature                     | Condition presents with a typical clinical or imaging feature.             | Diseases / Cancer → Symptoms / Pathological Features                                                                                                 | HT → Inhomogeneous echotexture   | feature_type, modality                               |
| measured_by                     | A parameter (source) is measured by a method (target). Source = parameter. | Hormones / Biomarkers / Lab Findings → Diagnostic Methods                                                                                            | anti-TPO titre → ELISA           | measurement_unit, threshold_value                    |
| administered_as                 | Dosage and administration details of a treatment.                          | Treatments → Treatments / Time Context                                                                                                               | Cholecalciferol → 60,000 IU/week | dosage, frequency, duration                          |
| compared_with                   | Comparison between two groups or interventions in a study.                 | Study Groups → Study Groups / Treatments → Treatments                                                                                                | Intervention group → Placebo     | comparator_role, study_endpoint                      |
| excluded_in_study               | Entities or conditions excluded from a study population.                   | Comorbidities / Diseases / Treatments → Study                                                                                                        | COVID-19 infection → PRECES study | reason_for_exclusion                                 |
| reported_in_study               | A claim or finding documented in a specific study.                         | Any clinical entity → Study                                                                                                                          | anti-TPO ↔ HT → Cooper2021       | sample_size, confidence_level                        |
| more_common_in                  | A condition appears more frequently in a specific group.                   | Diseases / Cancer / Symptoms → Patient Features & Demographics                                                                                       | Asthenia → Rural                 | comparative_group, statistical_significance          |
| has_an_incidence_peak_in        | A disease has a typical peak in age or life period.                        | Diseases / Cancer → Time Context / Patient Features & Demographics                                                                                   | PTL → Seventh Decade             | peak_period                                          |
| may_arise_from_the_setting_of   | A condition may develop in the context of another.                         | Cancer / Diseases → Diseases                                                                                                                         | PTL → HT                         | temporal_relationship                                |
| may_progress_to                 | One condition may evolve into another.                                     | Diseases / Cancer → Diseases / Cancer                                                                                                                | MALT → DLBCL                     | progression_probability                              |
| is_a_pathologic_feature_of      | A histological/microscopic feature typical of a disease.                   | Pathological Features → Diseases / Cancer                                                                                                            | Interstitial Lymphocytes → HT    | tissue_type                                          |
| is_a_type_of                    | The entity belongs to a broader category.                                  | Diseases / Cancer / Mechanisms → Diseases / Cancer / Mechanisms                                                                                      | DLBCL → Lymphoma                 | histologic_classification                            |
| is_a_symptom_of                 | A symptom belonging to a specific disease.                                 | Symptoms → Diseases / Cancer                                                                                                                         | Palpable Mass → PTL              | symptom_location                                     |
| can_present_with                | Common clinical presentation of a disease.                                 | Diseases / Cancer → Symptoms                                                                                                                         | PTL → B Symptoms                 | systemic_or_local                                    |
| follows                         | One event or treatment occurs after another.                               | Treatments / Medical Event → Treatments / Medical Event                                                                                              | RAI → Levothyroxine              | temporal_gap                                         |
| leads_to_remission_of           | A treatment leads to remission of a disease.                               | Treatments → Diseases / Cancer                                                                                                                       | Surgery → PTL                    | time_to_remission, durability                        |
| relapses_into                   | A condition returns after remission.                                       | Diseases / Cancer → Diseases / Cancer                                                                                                                | Remission → Hypothyroidism       | time_to_relapse, recurrence_rate                     |
| recommended_by_guideline        | A treatment or method officially recommended by a guideline.               | Treatments / Diagnostic Methods → Guideline                                                                                                          | Levothyroxine → ATA              | year, evidence_grade                                 |
| has_evidence_grade              | Strength of evidence for a claim or recommendation.                        | Treatments / Diagnostic Methods → Guideline (grade noted in key_properties)                                                                          | Levothyroxine → Grade A          | grade, grading_system                                |
| has_genetic_variant             | A gene carries a variant associated with disease risk or function.         | Gene → Genetic Variant                                                                                                                               | CTLA-4 → rs231775                | risk_direction, effect_size                          |
| alters_microbiota               | A condition or intervention shifts microbial abundance.                    | Diseases / Treatments / Lifestyle Factor → Gut Microbiota Taxon                                                                                      | HT → Firmicutes                  | direction, magnitude                                 |

## Extraction Rules

1. **Use only the listed `relation_type` values**
   1.1 The `relation_type` field MUST EXACTLY match one of the 32 snake_case names above. Lowercase, underscores between words. No spaces, no hyphens, no PascalCase.
   1.2 Do not invent new relation types.

2. **Direction matters**
   2.1 Each relation has a defined `source_type → target_type` direction. Do NOT reverse it.
   2.2 If a sentence implies the inverse direction (e.g., "X is detected by Y" vs "Y detects X"), match the canonical direction from the table.
   2.3 If neither direction matches the table, do not extract the relation.

3. **Only EXPLICITLY claimed relations**
   3.1 Relations must be claims directly made by the text — causal, diagnostic, statistical, or stated association.
   3.2 Methodological mentions ("X was recorded as a variable", "Y was tested using Z", "group A was defined as...") do NOT support a relation.
   3.3 Co-mention of two entities in the same sentence is NOT sufficient. The text must claim a specific relation between them.

4. **Materials & Methods sections — special caution**
   4.1 In sections describing study methodology, ONLY extract these relation types: `excluded_in_study`, `compared_with`, `diagnosed_by`, `is_a_serologic_marker_of`, `measured_by`.
   4.2 Skip all other relation types in methods sections — descriptions of procedures, variable definitions, and statistical tests are NOT relations.

5. **Source/target sourcing**
   5.1 `source_entity` and `target_entity` MUST be drawn from the pre-extracted entities list below.
   5.2 Do not invent entities. If a relation requires an entity not in the list, do not extract that relation.
   5.3 Copy each entity's `canonical_name` and `entity_type` EXACTLY as listed. Do not modify, abbreviate, expand, or invent variants.
   5.4 Pre-extracted entities also carry `surface_form` and `aliases` fields — use these to match how the text mentions an entity (e.g., text says "HT" and an entity has `surface_form: "HT"` / `canonical_name: "Hashimoto's Thyroiditis"`). Output ONLY `canonical_name` and `entity_type` in `source_entity` / `target_entity`.

6. **Evidence rules** (strict)
   6.1 `evidence` MUST be a single contiguous sentence copied verbatim from the text.
   6.2 Do NOT use ellipsis ("...") under any circumstances.
   6.3 Do NOT paraphrase, summarize, or stitch fragments from non-adjacent locations.
   6.4 If supporting context spans multiple sentences, choose the single most direct one.

8. **Negation and certainty rules**
   8.1 `claim_polarity` — choose one:
       - `positive`     — the relation is directly affirmed ("X is associated with Y", "X causes Y")
       - `negative`     — the relation is explicitly denied ("X was NOT associated with Y", "X failed to show an effect on Y")
       - `mixed`        — the text reports both supporting and contradicting evidence for the relation
       - `uncertain`    — the text is ambiguous or inconclusive ("the role of X in Y is unclear")
       - `hypothetical` — the relation is proposed as speculation or hypothesis ("X may be involved in Y")
   8.2 `claim_certainty` — choose one:
       - `high`     — directly stated finding, significant p-value, or strong causal claim
       - `moderate` — trend reported, borderline significance, or observational association
       - `low`      — speculation, single case, pilot data, or author opinion
   8.3 Do NOT convert negated findings into positive relations. These phrases signal `negative` polarity:
       - "no association", "not associated", "failed to show", "did not increase",
         "no significant difference", "no effect", "was not found to"
   8.4 These phrases signal `hypothetical` polarity:
       - "may", "might", "could", "possibly", "is thought to", "is hypothesized to", "remains to be confirmed"

7. **Output format** (strict)
   7.1 Return a single valid JSON object only — no extra prose, no commentary.
   7.2 Do NOT wrap the output in markdown code fences (no ```json).
   7.3 The output must be a single JSON object, not a list.
   7.4 All property names must be enclosed in double quotes.

## Output Example

{{
  "relations": [
    {{
      "relation_type": "is_a_serologic_marker_of",
      "source_entity": {{
        "entity_type": "Hormones, Biomarkers & Antibodies",
        "canonical_name": "Thyroid Peroxidase Antibody"
      }},
      "target_entity": {{
        "entity_type": "Diseases & Conditions",
        "canonical_name": "Hashimoto's Thyroiditis"
      }},
      "key_properties": {{
        "test_type": "antibody",
        "diagnostic_role": "primary serologic marker"
      }},
      "evidence": "Increased serum TPOAbs are used to confirm the diagnosis of Hashimoto's thyroiditis.",
      "claim_polarity": "positive",
      "claim_certainty": "high"
    }}
  ]
}}

## Pre-extracted Entities

{entities}

## Text for Analysis

'''
{text}
'''
"""
