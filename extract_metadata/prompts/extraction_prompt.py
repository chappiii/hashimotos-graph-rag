def get_extraction_prompt(text: str) -> str:
    return f"""
You are an expert at extracting metadata from academic papers. Your task is to fill in the values in the exact JSON format provided below, **strictly and solely based on the content of the academic paper text you will be given. Do not invent or infer any information not explicitly present or clearly derivable from the text.**

IMPORTANT: Do not use any thinking tags like <think></think> in your response. Provide ONLY the JSON output directly.

- **doi**: The Digital Object Identifier of the paper. Look for it typically near the top or bottom of the first page, often preceded by "doi:" or "https://doi.org/". If not found, use `null`.
- **title**: The full title of the academic paper. Usually found prominently at the top of the first page, often in a larger font size, bolded, or centered. **You MUST find the title; it is a fundamental part of any academic paper. If it's not immediately obvious, thoroughly scan the first few paragraphs or the very top section of the page.** It is highly improbable for a paper to lack a title; therefore, only use `null` as an absolute last resort if, despite rigorous scanning, no identifiable title is present in the provided text.
- **published_year**: The year the paper was officially published. Look for it near the publication details (e.g., citation, received/accepted/published dates). If not found, use `null`.
- **author_list**: A list of the full names of all authors. Extract names exactly as they appear in the author section.
Example: ["John Doe", "Jane Smith"]
- **countries**: A list of unique countries associated with the authors' affiliations. ***If an affiliation mentions a city, identify its corresponding country.*** Ensure each country appears only once in the list. If no country is found, use `[]`.
Example: ["USA", "Germany", "Japan"]
- **purpose_of_work**: A concise summary (20-40 words) explaining the main goal or objective of the research presented in the paper. Extract this from anywhere within the provided text, identifying the core reason or problem the paper addresses. If multiple relevant parts are found, synthesize them into a single, concise summary within the word limit. If the main goal cannot be clearly identified or summarized within 20-40 words, use `null`.
- **keywords**: A list of significant keywords that describe the paper's main topics. Prioritize extracting these directly from a dedicated "Keywords" section if one is present. **If no such section exists, extract 3 to 4 key terms or phrases that best represent the paper's main topics from the abstract and main text.** Ensure individual keywords are extracted correctly, even if they are separated by commas (,) or semicolons (;). If no suitable keywords are found, use `[]`.
Example: ["Advanced oxidation protein products", "Apoptosis", "Reactive oxygen species", "Hashimoto's thyroiditis"]

Return ONLY a valid JSON object (no explanation, no markdown, no comments). If any field is missing (i.e., not found explicitly or clearly derivable from the text based on the strict instructions above), use `null` (for string fields) or `[]` (for list fields). Do NOT change the provided keys or the structure of the JSON.

JSON Template:
{{
"doi": null,
"title": null,
"published_year": null,
"author_list": [],
"countries": [],
"purpose_of_work": null,
"keywords": []
}}

Text to analyze:
\"\"\"
{text}
\"\"\"
""" 