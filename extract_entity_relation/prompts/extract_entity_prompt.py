"""
Entity extraction prompt.

The Reference Vocabulary block is built once at module load from CANONICAL_TERMS
and substituted into the prompt template. The resulting `extract_entity_prompt`
is still a `.format()`-compatible template (only placeholder is `{text}`).
"""

from collections import defaultdict

from extract_entity_relation.canonical_terms import CANONICAL_TERMS


def _format_vocabulary(terms):
    by_type = defaultdict(list)
    for entry in terms:
        by_type[entry["type"]].append(entry)

    blocks = []
    for etype, entries in by_type.items():
        lines = [f"\n{etype}:"]
        for e in entries:
            if e["aliases"]:
                aliases_str = " | ".join(f'"{a}"' for a in e["aliases"])
                lines.append(f'- {aliases_str} -> "{e["canonical"]}"')
            else:
                lines.append(f'- "{e["canonical"]}"  (use this exact form; pin entity_type)')
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


_VOCABULARY = _format_vocabulary(CANONICAL_TERMS)


_PROMPT_TEMPLATE = """
You are a top-tier biomedical information extraction algorithm specialized in identifying clinical and scientific entities for knowledge graph construction.

Your task is to extract all relevant entities from the given text according to the Reference Vocabulary, Entity Schema, and Extraction Rules below.

For each extracted entity, return:
- `entity_type` — must EXACTLY match a label from the Schema (or the type pinned by Reference Vocabulary)
- `surface_form` — the exact verbatim text as it appears in the source
- `canonical_name` — the normalized biomedical name (from Reference Vocabulary if matched; otherwise the most specific name available)
- `normalized` — boolean: true if Reference Vocabulary lookup or in-chunk abbreviation expansion was applied; false otherwise
- `aliases` — list of alternative names mentioned IN THIS CHUNK only (use [] if none)
- `key_properties` — all properties defined for that entity type (use `null` for unknown values; do NOT omit fields)
- `evidence` — a single verbatim sentence from the text supporting the extraction

## Reference Vocabulary

When a surface form in the text matches any alias below (case-insensitive; apostrophe and dash variants count as the same character), you MUST:
- Set `canonical_name` to the listed canonical form
- Set `entity_type` to the listed type — even if context would suggest a different label
- Set `normalized` to true

For entries shown without aliases, the canonical form itself is the only surface form; still pin the entity_type as listed.
__VOCABULARY__

## Entity Schema

| Entity Type                          | Description                                                                | Examples                                                                             | Key Properties                                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| Diseases & Conditions                | Non-cancerous health conditions and diseases, often chronic or autoimmune. | Hashimoto's Thyroiditis, Autoimmune Thyroid Disease, Hypothyroidism, Obesity         | Disease name, Autoimmune (Boolean), Chronicity (chronic/acute), Organ/System affected, Endocrine-related (Boolean),                  |
| Cancer Types / Malignancies          | Malignant neoplasms, mostly lymphomas or thyroid-related cancers.          | Diffuse Large B-cell Lymphoma, MALT Lymphoma, Hodgkin Lymphoma, Thyroid Malignancies | Cancer subtype, Cell lineage (B-cell/T-cell), Aggressiveness (indolent/aggressive), Primary site, Histological subtype,              |
| Symptoms & Clinical Findings         | Observed symptoms and clinical presentation features.                      | Fatigue, Weight gain, Palpable Mass in Neck, Hoarseness, B Symptoms                  | Symptom name, Symptom category (general/systemic/local), Duration (acute/chronic), Severity (if known), Location (if applicable),    |
| Hormones, Biomarkers & Antibodies    | Substances measured in blood or tissues for diagnosis or monitoring.       | TSH, Anti-TPO, FT4, Vitamin D, Triglycerides                                         | Molecule name, Type (hormone/antibody/vitamin/lipid), Associated condition, Reference range, Direction of abnormality (e.g., ↑ TSH), |
| Diagnostic Methods & Criteria        | Tools, measurements, and criteria used to diagnose diseases.               | Thyroid ultrasonography, Echogenicity, Clinical history of HT                        | Method name, Imaging modality, Measurement criteria (e.g., size > 5 mm), Interpretation characteristics, Associated disease,         |
| Pathological & Histological Features | Microscopic features observed in biopsies or pathology slides.             | Interstitial Lymphocyte Infiltrate, MALT-type Component                              | Feature name, Tissue type, Cell composition, Pattern type (e.g., follicular, infiltrative), Diagnostic relevance,                    |
| Molecular & Immune Mechanisms        | Molecular/cellular pathways and immune mechanisms involved in disease.     | T cells, B cells, IFN-γ, NF-κB Pathway, Chronic B-cell Stimulation                   | Molecule/Cell type, Pathway name, Immune mechanism type, Role in disease (e.g., stimulation, suppression), Genetic association,      |
| Patient Features & Demographics      | Descriptive or demographic data about patients.                            | Age, Sex, Seventh Decade, Urban/Rural residence                                      | Age (numerical or range), Sex, Residence type, Incidence peak (e.g., Seventh Decade), Eligibility criteria (e.g., Age > 18),         |
| Comorbidities & Risk Factors         | Conditions or factors increasing disease likelihood or co-occurring.       | Depression, Cardiovascular disease, COVID-19, Vitamin D deficiency                   | Risk factor name, Comorbidity type (physical/mental), Relevance to primary disease, Modifiability (modifiable/non-modifiable),       |
| Treatments & Management              | Interventions used to manage or cure conditions.                           | Levothyroxine, Surgery, Chemo/Radiotherapy, Rituximab                                | Treatment type (pharmacological/surgical), Drug name, Therapy class, Treatment target (e.g., hormone replacement), Usage indication, |
| Laboratory Findings                  | Quantitative or qualitative lab test results.                              | TSH > 5.0 mIU/L, Vitamin D < 20 ng/ml, Anti-TPO antibody level                       | Test name, Value, Unit, Reference range, Interpretation (elevated/decreased), Test type (hormone/antibody/vitamin),                  |
| Study Groups                         | Groups defined in a clinical study for comparison or intervention.         | Cholecalciferol group, Placebo group                                                 | Group name, Treatment/intervention type, Role (control/intervention), Sample size (if available), Study arm assignment,              |
| Time Context                         | Clinical timelines and durations                                           | "Chronic", "3 months", "Seventh Decade"                                              | Time span (e.g., acute/chronic), Duration (e.g., 3 months), Temporal qualifier (e.g., early/late)                                    |
| Lifestyle Factor                     | Diet, exercise, stress, habits                                             | High iodine diet, smoking                                                            | Factor type (e.g., diet, habit), Frequency/intensity, Duration, Impact on health (if known)                                          |
| Environmental Factor                 | Radiation, pollution, geographic risks                                     | Fukushima exposure, high-altitude                                                    | Factor source (e.g., location, substance), Exposure level, Duration, Known associated conditions                                     |
| Medical Event                        | Events or states within clinical history                                   | Recurrence, Surgery, Remission                                                       | Event type (e.g., procedure, relapse), Timing (onset/offset), Outcome, Associated condition                                          |
| Study Design                         | Study design classification                                                | RCT, Meta-analysis, Cohort                                                           | Design type, Evidence strength level, Sample size (if known), Blinding/randomization info                                            |
| Access Type                          | Treatment accessibility                                                    | OTC, prescription-only                                                               | Access level (OTC, Rx, restricted), Regulatory region (e.g., FDA, EMA), Cost/access barriers                                         |
| Gene                                 | Genes implicated in HT susceptibility, thyroid function, or autoimmunity.  | HLA-DR3, CTLA-4, PTPN22, FOXP3, TG                                                   | Gene symbol, Chromosome (if known), Known function, Associated condition, Expression tissue                                          |
| Genetic Variant                      | SNPs, alleles, or mutations linked to HT risk or thyroid function.         | rs2476601, HLA-DR3 allele, FOXP3 variant                                             | Variant ID (rsID/allele), Variant type (SNP/CNV/allele), Parent gene, Risk direction (risk/protective), Effect size (OR/β if known)  |
| Gut Microbiota Taxon                 | Bacterial taxa associated with HT or thyroid function.                     | Bacteroidetes, Firmicutes, Lactobacillus, Prevotella                                 | Taxon name, Taxonomic level (phylum/genus/species), Abundance direction (enriched/depleted in HT), Sample source (stool/oral)        |
| Study                                | A research study cited as the evidence source for a claim.                 | Cooper et al. 2021, NHANES III, Mendelian randomization analysis                     | Study ID/citation, Study type (RCT/cohort/MR/meta-analysis), Sample size, Year, Authors, Journal                                     |
| Guideline                            | Clinical practice guidelines or consensus statements.                      | ATA 2023, NICE NG145, ETA guideline                                                  | Guideline name, Issuing body (ATA/NICE/ETA), Year, Recommendation strength, Target condition                                         |

## Extraction Rules

1. **Schema adherence**
   1.1 Only extract entities whose `entity_type` matches a label in the Schema EXACTLY (or is pinned by Reference Vocabulary). Do not invent new types.
   1.2 All `key_properties` listed for an entity type must appear. Use `null` for unknown values; do not omit fields.

2. **Grounding**
   2.1 Values must be explicitly stated in the text OR clearly inferable by simple logic/math (e.g., complementary group percentages summing to 100%).
   2.2 If an abbreviation is NOT explicitly defined or unambiguous in the chunk (e.g., "AS" appearing without context), DO NOT extract it.
   2.3 If you are uncertain about an entity's `entity_type`, DO NOT extract it. Better to omit than to misclassify.

3. **Evidence rules** (strict)
   3.1 `evidence` MUST be a single contiguous sentence copied verbatim from the text.
   3.2 Do NOT use ellipsis ("...") under any circumstances.
   3.3 Do NOT paraphrase, summarize, or stitch fragments from non-adjacent locations.
   3.4 If supporting context spans multiple sentences, choose the single most direct one.

4. **Numerical handling**
   4.1 If a percentage is given with a total (e.g., "49.2%" and "N=120"), include both the percentage and the absolute count.
   4.2 For two complementary groups, you may infer the second group's value by subtraction.
   4.3 Preserve all statistics verbatim — means, SD, p-values, units (e.g., "47.1 ± 14.8 y", "p = 0.426"). Do not round.

5. **Output format** (strict)
   5.1 Return a single valid JSON object only — no extra prose, no commentary.
   5.2 Do NOT wrap the output in markdown code fences (no ```json).
   5.3 The output must be a single JSON object, not a list.
   5.4 All property names must be enclosed in double quotes.

6. **Normalization rules**
   6.1 `surface_form` — the exact verbatim text as written in the chunk (preserve original case, punctuation, hyphens, apostrophes).
   6.2 `canonical_name`:
       - If `surface_form` matches an alias in the Reference Vocabulary, use the listed canonical form.
       - Otherwise, use the most specific biomedical name available. Expand abbreviations ONLY if the expansion is clearly defined within this same chunk (e.g., chunk says "interleukin-6 (IL-6)" — canonical = "Interleukin-6").
       - If no normalization applies, set `canonical_name` equal to `surface_form`.
   6.3 `entity_type`:
       - If the canonical name is in Reference Vocabulary, use the vocabulary's entity_type.
       - Otherwise, choose the best fit from the Schema.
   6.4 `normalized` — true if rule 6.2 applied a vocabulary lookup or an in-chunk expansion; false otherwise.
   6.5 `aliases` — list alternative names that appear IN THIS CHUNK and are NOT redundant with `canonical_name` or `surface_form`. Example: chunk says "Hashimoto's thyroiditis (HT) is..." → for the HT entity, `aliases: ["HT"]`. Do NOT echo Reference Vocabulary aliases. Use [] if no other names appear in the chunk.

## Output Example

{{
  "entities": [
    {{
      "entity_type": "Diseases & Conditions",
      "surface_form": "Hashimoto's thyroiditis",
      "canonical_name": "Hashimoto's Thyroiditis",
      "normalized": true,
      "aliases": ["HT", "Hashimoto thyroiditis"],
      "key_properties": {{
        "Disease name": "Hashimoto's Thyroiditis",
        "Autoimmune": true,
        "Chronicity": "chronic",
        "Organ/System affected": "thyroid gland",
        "Endocrine-related": true
      }},
      "evidence": "Hashimoto's thyroiditis (HT) is a chronic autoimmune disease of the thyroid gland that frequently progresses to hypothyroidism."
    }}
  ]
}}

## Input Text

'''
{text}
'''
"""


# Substitute the vocabulary at module load. Using a non-brace placeholder so it
# does not interfere with `.format(text=...)` at call time.
extract_entity_prompt = _PROMPT_TEMPLATE.replace("__VOCABULARY__", _VOCABULARY)
