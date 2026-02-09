def get_correction_prompt(llm_response: str) -> str:
    return f"""You are a meticulous proofreader. Your task is to correct only spelling errors and remove any excess spaces from the provided text. Do not change any other part of the text, its structure, or its meaning. For author_list names, you may only fix spacing and punctuation, but must NOT change, add, or remove letters. Return the corrected text exactly as it was given, with only the specified edits. Ensure each country appears only once in the list, even if it's mentioned multiple times.
Return ONLY a valid JSON object (no explanation, no markdown, no comments).
Text to correct:
{llm_response}
""" 