extract_relation_prompt = """
You are a biomedical expert specializing in relationship extraction.

Your task is to identify and extract relationships between the provided list of entities, based on the given text.

**Instructions:**

1.  Analyze the text to find connections between the entities listed in the **"Pre-extracted Entities"** section.
2.  For each relationship, define the `source_entity`, `target_entity`, and the `relation_type`.
3.  The relation_type must match one of the types listed in the table below.
4.  Provide the exact `evidence` sentence from the text that supports the relationship.
5.  Return the output in the specified JSON format.
***Relationships table : ***

| Relationship Type                     | Description                                                           | Example                                                      | Key Properties                                                                                                      |
| ------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| causes / increases risk of            | One entity causes or increases the likelihood of another.             | HT → PTL, TSH > 4.5 → Hypothyroidism                         | Source entity, Target entity, Risk strength (if known), Direction (cause/risk), Evidence type (e.g., clinical, lab) |
| associated with                       | Statistically or clinically meaningful association without causality. | Vitamin D deficiency ↔ HT                                    | Entity A, Entity B, Association strength (if known), Type of association (clinical, statistical)                    |
| negatively correlated with            | An inverse correlation between two variables.                         | 25(OH)D ↔ anti-TPO                                           | Variable A, Variable B, Correlation coefficient (if known), Direction = negative                                    |
| positively correlated with            | A positive correlation between two variables.                         | TgAb ↔ Triglycerides                                         | Variable A, Variable B, Correlation coefficient (if known), Direction = positive                                    |
| treated with / managed by             | A treatment or intervention used to manage a condition.               | HT → Vitamin D supplementation                               | Disease, Treatment, Treatment type (e.g., pharmacological), Effectiveness (if known)                                |
| improves / reduces                    | An intervention reduces or improves a clinical/biological parameter.  | Cholecalciferol → reduces → anti-TPO antibody                | Intervention, Target parameter, Direction of effect, Percent/degree of change (if known)                            |
| has therapeutic effect on             | A substance with direct therapeutic effect on a condition.            | Vitamin D → HT                                               | Therapeutic agent, Target condition, Mechanism (if known), Mode of action                                           |
| inhibited by                          | A mechanism or process suppressed by another agent.                   | 1,25(OH)₂D → inhibits B cell proliferation                   | Inhibitor, Target process, Immune component, Suppression mechanism                                                  |
| induces                               | A biological mechanism triggers another process.                      | IFN-γ → HLA-DR expression                                    | Inducer, Induced feature, Cell type involved, Pathway/mechanism                                                     |
| diagnosed by                          | Diagnostic method, marker, or criterion used to identify a condition. | HT → Elevated anti-TPO                                       | Disease, Diagnostic tool, Modality type (e.g., serologic, imaging), Diagnostic criteria                             |
| is a diagnostic method for            | A tool or marker specifically used to diagnose a disease.             | Antithyroid Antibodies → HT                                  | Method, Target disease, Modality, Diagnostic specificity (if known)                                                 |
| co-occurs with                        | Two conditions or findings commonly present together.                 | HT ↔ Asthenia / Headache                                     | Co-occurring entities, Clinical relevance, Population/subgroup                                                      |
| more common in                        | A condition appears more frequently in a specific group.              | Asthenia → Rural                                             | Condition, Comparative group, Reference group, Statistical significance (if known)                                  |
| excluded in study                     | Entities or conditions excluded from a study population.              | COVID-19 infection → exclusion                               | Excluded factor, Study context, Reason for exclusion                                                                |
| detectable by                         | A finding or feature detectable using a method or modality.           | Thyroid nodule → Ultrasound                                  | Target finding, Detection method, Imaging/lab modality                                                              |
| has feature                           | A condition presenting with a typical clinical or imaging feature.    | HT → Inhomogeneous echotexture                               | Condition, Feature type, Modality (e.g., ultrasound), Diagnostic value                                              |
| measured by                           | A parameter measured by a specific lab or imaging method.             | Thyroid autoimmunity → anti-TPO titre                        | Parameter, Measurement tool, Measurement unit, Threshold value (if known)                                           |
| administered as                       | Dosage and administration details of a treatment.                     | Cholecalciferol → 60,000 IU/week                             | Drug, Dosage, Frequency, Duration                                                                                   |
| compared with                         | Comparison between two groups or interventions in a study.            | Intervention group → Placebo                                 | Group A, Group B, Comparator role (intervention/control), Study endpoint (if known)                                 |
| may arise from the setting of         | A condition may develop in the context of another.                    | PTL → may arise from HT                                      | Secondary condition, Primary setting, Temporal relationship (chronicity)                                            |
| may progress to                       | One condition may evolve into another.                                | MALT → DLBCL                                                 | Initial condition, Progressed condition, Progression probability (if known)                                         |
| is a subtype of                       | A disease or finding is a subtype of another category.                | DLBCL → PTL                                                  | Subtype, Parent type, Histologic classification                                                                     |
| is a type of                          | The entity belongs to a broader category.                             | PTL → Thyroid Malignancy                                     | Specific entity, General category                                                                                   |
| is a symptom of                       | A symptom belonging to a specific disease.                            | PTL → Palpable Mass                                          | Symptom, Underlying disease, Symptom location (if applicable)                                                       |
| can cause                             | A finding or condition may lead to other symptoms.                    | Mass → Dysphagia, Hoarseness                                 | Cause feature, Resulting symptoms, Mechanism (e.g., compression)                                                    |
| can present with                      | Common clinical presentation of a disease.                            | PTL → B Symptoms                                             | Disease, Symptom set, Systemic or local classification                                                              |
| has higher prevalence of HT than      | A group has a higher rate of HT than another.                         | MALT Lymphoma → DLBCL                                        | Group A, Group B, Comparative prevalence                                                                            |
| is a pathologic feature of            | A histological or microscopic feature typical of a disease.           | Interstitial Lymphocytes → HT                                | Pathologic feature, Associated disease                                                                              |
| is an etiologic factor in             | A cause or trigger in the pathogenesis of a disease.                  | Chronic B-cell stimulation → Genetic Events                  | Etiologic factor, Target condition, Pathogenic mechanism                                                            |
| plays a critical role in pathogenesis | A mechanism central to the development of disease.                    | NF-κB Activation → PTL                                       | Mechanism, Disease, Functional consequence                                                                          |
| may not derive from                   | The condition may not originate from the assumed context.             | Subset of DLBCL → Not from MALT                              | Target entity, Absent origin, Implication                                                                           |
| is a serologic marker of              | A biomarker or antibody is used to identify a disease.                | Anti-TPO → HT                                                | Marker, Target disease, Test type (e.g., antibody), Diagnostic role                                                 |
| occurs preferentially in              | A condition occurs more frequently in a demographic group.            | PTL → Females                                                | Condition, Demographic group                                                                                        |
| has an incidence peak in              | A disease has a typical peak in age or life period.                   | PTL → Seventh Decade                                         | Disease, Peak incidence age/group                                                                                   |
| follows                               | One event or treatment occurs after another.                          | RAI → follows → Levothyroxine                                | Preceding event, Following event, Temporal gap (if known), Clinical context                                         |
| precedes                              | One event occurs before another.                                      | Hyperthyroidism → precedes → Thyroidectomy                   | Earlier event, Later event, Sequence certainty, Supporting evidence                                                 |
| leads to remission of                 | A treatment leads to remission of a disease.                          | Surgery → remission → PTL                                    | Treatment type, Target condition, Time to remission, Remission durability                                           |
| is over the counter                   | A drug or treatment is available without prescription.                | Cholecalciferol → is_over_the_counter → TRUE                 | Substance name, Regulatory status, Region (e.g., US, EU), Access limitations                                        |
| is first line treatment for           | A treatment is recommended as first-line therapy.                     | Levothyroxine → is_first_line_treatment_for → Hypothyroidism | Treatment name, Indicated condition, Guideline name/source, Strength of recommendation                              |
| recommended by guideline              | A treatment is officially recommended by a guideline.                 | Levothyroxine → recommended_by_guideline → ATA               | Treatment, Guideline authority (e.g., ATA, NICE), Year/version, Evidence grade (if available)                       |
| based on small sample                 | A study has sample size limitations.                                  | P23 → based_on_small_sample → TRUE                           | Study ID, Sample size, Statistical power, Acknowledged limitation (Yes/No), Impact on conclusions                   |
| relapses into                         | A condition returns after remission.                                  | Remission → relapses_into → Hypothyroidism                   | Initial condition, Relapsed condition, Time to relapse, Risk factors (if known), Recurrence rate                    |
---

{entities}

**Text for Analysis:**
'''
{text}
'''

**Output Format:**

```json
{{
  "relations": [
    {{
      "relation_type": "DIAGNOSED_WITH",
      "source_entity": {{
        "entity_type": "Patient Features & Demographics",
        "canonical_name": "45-year-old female"
      }},
      "target_entity": {{
        "entity_type": "Diseases & Conditions",
        "canonical_name": "Hashimoto's thyroiditis"
      }},
      "key_properties": {{
        "certainty_level": "confirmed"
      }},
      "evidence": "The patient, a 45-year-old female, was diagnosed with Hashimoto's thyroiditis."
    }}
  ]
}}
```
"""
