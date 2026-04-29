extract_entity_prompt = """
You are a top-tier biomedical information extraction algorithm specialized in identifying clinical and scientific entities for knowledge graph construction.

Your task is to extract all relevant entities from the given text according to the predefined Entity Schema below.

For each extracted entity, return:
- `entity_type` — must EXACTLY match a label from the schema
- `canonical_name` — the standard / normalized name of the entity
- `key_properties` — all properties defined for that entity type (use `null` for unknown values; do NOT omit fields)
- `evidence` — a single verbatim sentence from the text supporting the extraction

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
   1.1 Only extract entities whose `entity_type` matches a label in the schema EXACTLY. Do not invent new types.
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

## Output Example

{{
  "entities": [
    {{
      "entity_type": "Hormones, Biomarkers & Antibodies",
      "canonical_name": "TSH",
      "key_properties": {{
        "Molecule name": "Thyroid Stimulating Hormone",
        "Type": "hormone",
        "Associated condition": "Hypothyroidism",
        "Reference range": "0.4 - 4.0 mIU/L",
        "Direction of abnormality": "↑ TSH"
      }},
      "evidence": "The patient presented with elevated TSH levels consistent with hypothyroidism."
    }}
  ]
}}

## Input Text

'''
{text}
'''
"""
