from __future__ import annotations

CAST_SYSTEM = """You are designing a cast-level voice matrix.
Hard constraints:
- Prevent voice convergence across the cast.
- Avoid author-default phrasing.
- Each character must have a distinct social mask + rhetorical habits.
Output must be YAML only. No backticks.
"""

CAST_VOICE_MATRIX_PROMPT = """Given the plot and the cast list, produce a CAST VOICE MATRIX.

For EACH character, output:
- do: 6–10 bullets (what they reliably do in speech)
- avoid: 6–10 bullets (rhetorical moves they do NOT do)
- banned_phrases: 10–20 phrases/words they will not say (include filler words and pet phrases)
- banned_metaphor_domains: 3–6 domains (e.g. sports, military, therapy, tech, religion, finance)
- syntax_rules: 5–10 rules (sentence length, punctuation habits, interruptions, contractions, questions, fragments)
- social_mask: 1–2 sentences describing the persona they project in public

Also output:
- global_banned_habits: 10 bullets that NONE of the cast should overuse (prevents samey "author voice").

Rules:
- No two characters may share more than 30% overlap in banned_phrases or metaphor bans.
- If two characters are similar by role, differentiate via worldview and social mask.

Return YAML with keys:
- global_banned_habits
- characters (mapping slug -> voice_constraints object)
"""
