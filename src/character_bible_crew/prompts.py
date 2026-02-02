from __future__ import annotations

SYSTEM_RULES = """You are part of a character-bible team.
Hard constraints:
- Do NOT contradict the provided plot facts.
- If the plot does not specify something, you may propose options, but label them clearly as options or open questions.
- Avoid generic, author-voice defaulting. Prefer specific, situationally-grounded details.
- Minimize tropes unless they are intentionally used AND given a fresh angle.
Output constraints:
- Return JSON only when asked for JSON.
- Use concise but vivid language; no purple fog.
"""

DIFFERENTIATOR_PROMPT = """Create a Differentiation Signature for the character.

Goal: prevent 'author clone' syndrome.

Produce:
1) 5 bullets: "This character is NOT the author because..."
2) 5 bullets: distinctive worldview assumptions (what they believe is true about people/life)
3) 5 bullets: speech/voice constraints (e.g., sentence length, favorite metaphors, taboo phrases)
4) 3 contradiction pairs (e.g. "craves belonging / fears being known")

Return as JSON with keys:
- not_the_author_because
- worldview_assumptions
- voice_constraints
- contradictions

Return JSON only. Do not use triple backticks.
"""

OUTER_LAYER_PROMPT = """Fill the OUTER LAYER sections of the Reedsy template (Basics, Physical Appearance, Speech/Communication).

Be concrete: physicality, mannerisms, speech tempo, eye contact, posture, grooming, etc.
If unknown, propose 2–3 options and label as options.

Return JSON with keys:
- basics
- physical_appearance
- speech_and_communication

Return JSON only. Do not use triple backticks.
"""

FLESH_PROMPT = """Fill THE FLESH sections of the Reedsy template (Past, Family, External Relationships).

Make backstory causal: show how environment shaped defaults. Avoid melodrama unless plot supports it.
Include at least 2 "life beats" that create both a strength and a scar.

Return JSON with keys:
- past
- family
- external_relationships

Return JSON only. Do not use triple backticks.
"""

CORE_PROMPT = """Fill THE CORE sections of the Reedsy template (Psychology, Present/Future).

Make motivations specific to the plot, not generic self-help slogans.
Include moral-compromise conditions (when they bend their compass).
If details are unknown, label as open questions or options.

Return JSON with keys:
- psychology
- present_and_future

Return JSON only. Do not use triple backticks.
"""

DEVILS_ADVOCATE_PROMPT = """Your job is to be annoying in a useful way.

Given the current assembled profile, identify:
- Where it reads generic
- Where it feels like the author
- Where it duplicates other characters
- Where voice is indistinct

Then propose concrete fixes:
- Swap a belief
- Add a boundary/line they won't cross
- Give them a weird competence or blind spot
- Change the social mask they wear

Return Markdown with:
- Clone risks (bullets)
- Fixes (bullets)
- 3 "stress tests": short scenario prompts that would reveal their uniqueness in dialogue/action
"""

FINALIZER_PROMPT = """You are the Showrunner/Finalizer.

You will receive:
- Plot facts
- Character brief
- Differentiation signature
- Outer layer JSON
- Flesh JSON
- Core JSON
- Devil's advocate notes

Task:
1) Merge into ONE unified JSON following the CharacterReedsyProfile schema.
2) Resolve conflicts. If unresolved, add to notes.open_questions.
3) Preserve differentiation_signature and clone_risks_and_fixes.
4) Ensure it matches the Reedsy template categories closely.

Return JSON ONLY.
DO NOT wrap in triple backticks.
DO NOT include explanations, headings, or prose."""
