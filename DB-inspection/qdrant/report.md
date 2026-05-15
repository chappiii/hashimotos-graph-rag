
============================================================
1. OVERVIEW
============================================================

papers:
  points:          115
  vector dim:      3072  distance: Cosine
  indexed fields:  ['year', 'study_design', 'keywords', 'paper_id']
  payload keys:    ['paper_id', 'title', 'doi', 'year', 'study_design', 'study_weight', 'countries', 'keywords', 'purpose_of_work']

chunks:
  points:          1,765
  vector dim:      3072  distance: Cosine
  indexed fields:  ['paper_id', 'section_type']
  payload keys:    ['paper_id', 'section_name', 'section_type', 'chunk_filename', 'window_index', 'window_count', 'text']

evidence:
  points:          5,524
  vector dim:      3072  distance: Cosine
  indexed fields:  ['claim_polarity', 'claim_certainty', 'relation_type', 'paper_id']
  payload keys:    ['claim_signature', 'paper_id', 'section_type', 'claim_polarity', 'claim_certainty', 'relation_type', 'source_entity', 'target_entity', 'evidence_text']

============================================================
2. PER-PAPER COVERAGE
============================================================

chunks:
  papers covered:  115 / 115
  total points:    1,765
  avg per paper:   15.3
  min / max:       1 / 52

evidence:
  papers covered:  115 / 115
  total points:    5,524
  avg per paper:   48.0
  min / max:       7 / 118

============================================================
3. CHUNK SECTION_TYPE DISTRIBUTION
============================================================
  OTHER             640  ##############
  DISCUSSION        315  #######
  RESULTS           248  #####
  INTRODUCTION      186  ####
  METHODS           178  ####
  ABSTRACT          110  ##
  CONCLUSION         88  #

============================================================
4. EVIDENCE POLARITY & CERTAINTY BREAKDOWN
============================================================

Polarity:
  positive         5026
  negative          157
  mixed              11
  uncertain          43
  hypothetical      287

Certainty:
  high             4005
  moderate         1337
  low               182

============================================================
5. SAMPLE RECORDS
============================================================

--- papers (3 samples) ---
  [1] A Real-Life Study in Patients Newly Diagnosed with Autoimmune Hashimoto’s Thyroi
       study_design=cohort  year=2024
       purpose: To analyze the thyroid panel in newly diagnosed Hashimoto’s thyroiditis patients to assess the relationship between asth...
  [2] A Prospective Study to Evaluate the Possible Role of Cholecalciferol Supplementa
       study_design=rct  year=2022
       purpose: To assess the therapeutic role of Vitamin D in managing Hashimoto's Thyroiditis...
  [3] Hashimoto's thyroiditis-related myopathy in a patient with SARS-CoV-2 infection:
       study_design=case_report  year=2023
       purpose: This study investigates the link between SARS-CoV-2 infection and Hashimoto's thyroiditis-related myopathy, emphasizing ...

--- chunks (3 samples) ---
  paper=1  ABSTRACT  win 0/0  [1-abstract.md]
  text: **Abstract: Background:** Amid the large panel of autoimmune thyroid diseases, Hashimoto’s thyroiditis (HT) represents a major point across multidisciplinary da...
  paper=1  INTRODUCTION  win 0/1  [2-1._introduction.md]
  text: 1. Introduction Amid the large panel of autoimmune thyroid diseases, chronic Hashimoto’s thyroiditis (HT) represents a major point in daily practice from a mult...
  paper=1  INTRODUCTION  win 1/1  [2-1._introduction.md]
  text: recent COVID-19 pandemic highlighted many prior known or unknown medical and surgical entities and unexpected outcomes, including in the endocrine field [10–12]...

--- evidence (3 samples) ---
  paper=52  positive/moderate
  Acute Kidney Injury --[associated_with]--> Fibrillary Glomerulonephritis
  "The patient’s acute kidney injury was associated to hypovolemia with an underlying glomerulonephritis, and the patient was advised to increase fluid intake."
  paper=58  positive/high
  Thyroid Peroxidase Antibody --[associated_with]--> Pregnancy
  "TPOAbs are associated with the risk of hypothyroidism during pregnancy, increased likelihood of miscarriage, and failure of in vitro fertilization.44"
  paper=36  positive/moderate
  Thyroid Peroxidase Antibody --[is_a_serologic_marker_of]--> Autoimmune Thyroid Disease
  "This association was further supported by Shobha et al., who found that 25% of SLE patients had elevated AbTPO levels, indicating a high prevalence of AITD, par"

============================================================
6. SEMANTIC SIMILARITY TEST
============================================================
Query: Vitamin D supplementation effect on anti-TPO antibodies in Hashimoto's thyroiditis

--- top 3 from papers ---
  0.938  [29] Can Supplementation with Vitamin D Modify Thyroid Autoantibodies (Anti-TPO Ab, A
  0.923  [2] A Prospective Study to Evaluate the Possible Role of Cholecalciferol Supplementa
  0.871  [25] Autoimmune Thyroiditis and Vitamin D
--- top 3 from chunks ---
  0.905  paper=2  ABSTRACT
         **Introduction :** Several studies have reported a low Vitamin D status in Autoimmune Thyroid Diseases (AITD), indicating association betwee...
  0.895  paper=29  ABSTRACT
         ABSTRACT Hashimoto’s thyroiditis (HT) is the most prevalent autoimmune disorder characterized by the destruction of thyroid cells caused by ...
  0.884  paper=2  INTRODUCTION
         Hashimoto Thyroiditis (HT), an autoimmune disease in which thyroid cells are destroyed by antibody and cell-mediated immune processes. Hashi...
--- top 3 from evidence ---
  0.916  paper=2  Hashimoto's Thyroiditis -> Vitamin D Supplementation
         "Vitamin D can be a therapeutic option in Hashimoto's Thyroiditis."
  0.909  paper=2  Vitamin D Supplementation -> Thyroid Peroxidase Antibody
         "If supplementation of Vitamin D decreases anti-TPO antibody titres, in future it may become a part of AITDs' treatment, especially in those "
  0.906  paper=2  Vitamin D -> Thyroid Peroxidase Antibody
         "The 8 weeks randomized; double-blind, placebo-controlled clinical trial demonstrates a negative correlation between Serum 25 hydroxy Vitamin"

============================================================
7. QUALITY CHECKS
============================================================
payload completeness (papers — scrollable, small payloads):
  title                115/115  (100%)
  doi                  109/115  (95%)
  year                 115/115  (100%)
  study_design         115/115  (100%)
  countries            115/115  (100%)
  keywords             115/115  (100%)
  purpose_of_work      115/115  (100%)

point counts match expected:
  papers:   115 / 115 expected
  chunks:   1765 / 1765 expected
  evidence: 5524 / 5524 expected