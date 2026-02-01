from __future__ import annotations

from typing import Dict, Any, List

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from .llm import get_llm_config
from .prompts import (
    SYSTEM_RULES,
    DIFFERENTIATOR_PROMPT,
    OUTER_LAYER_PROMPT,
    FLESH_PROMPT,
    CORE_PROMPT,
    DEVILS_ADVOCATE_PROMPT,
    FINALIZER_PROMPT,
)

def _make_agent(role: str, goal: str, backstory: str, llm_kwargs: dict) -> Agent:
    # CrewAI versions differ in LLM plumbing. Many accept `llm=...` where ... can be a string or config.
    # We'll pass a dict; if your version wants a string, swap llm=llm_kwargs["provider_model"].
    ollama_llm = LLM(
        model="ollama/llama3",
        base_url="http://localhost:11434",
    )


    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        allow_delegation=False,
        verbose=True,
        llm=ollama_llm,
        system=SYSTEM_RULES,
    )


def build_character_crew(plot: str, roster_context: str, character: dict, cast_voice_matrix: dict | None = None) -> Crew:

    llm_cfg = get_llm_config()

    name = character["name"]
    role = character.get("role", "")
    brief = character.get("brief", "")
    gender = character.get("gender", {})
    orientation = character.get("orientation", {})
    slug = character["slug"]
    voice_for_this = cast_voice_matrix.get("characters", {}).get(slug, {})
    global_bans = cast_voice_matrix.get("global_banned_habits", [])

    # Agents
    differentiator = _make_agent(
        role="Character Differentiator",
        goal="Make this character unmistakably distinct from the author and from other characters.",
        backstory="You specialize in anti-clone design, voice constraints, and worldview differentiation.",
        llm_kwargs=llm_cfg,
    )

    outer = _make_agent(
        role="Outer Layer Specialist",
        goal="Create concrete exterior traits, mannerisms, and communication style.",
        backstory="You build vivid physical/sensory characterization without cliches.",
        llm_kwargs=llm_cfg,
    )

    flesh = _make_agent(
        role="Backstory & Relationships Specialist",
        goal="Build causal backstory, family context, and social dynamics aligned to plot.",
        backstory="You think in life beats, environments, and relationship physics.",
        llm_kwargs=llm_cfg,
    )

    core = _make_agent(
        role="Core Psychology Specialist",
        goal="Define inner drivers, fears, moral compromises, and story goal/motivation.",
        backstory="You translate theme into psychology without therapy-babble.",
        llm_kwargs=llm_cfg,
    )

    devil = _make_agent(
        role="Devil’s Advocate",
        goal="Attack the draft for sameness and propose fixes that create uniqueness.",
        backstory="You are allergic to generic protagonists and identical voices.",
        llm_kwargs=llm_cfg,
    )

    finalizer = _make_agent(
        role="Showrunner / Finalizer",
        goal="Merge outputs into one Reedsy-structured YAML; resolve conflicts; preserve canon.",
        backstory="You enforce plot truth and continuity across the cast.",
        llm_kwargs=llm_cfg,
    )

    # Shared context for tasks
    base_context = f"""PLOT FACTS:
{plot}

CANONICAL CHARACTER FACTS (DO NOT CONTRADICT):
- Gender identity: {gender.get("identity", "unspecified")}
- Pronouns: {gender.get("pronouns", "unspecified")}
- Romantic orientation: {orientation.get("romantic", "unspecified")}
- Sexual orientation: {orientation.get("sexual", "unspecified")}
- Orientation notes: {orientation.get("notes", "")}

If any of the above are 'unknown' or 'not decided', do NOT invent. Instead, add open questions.

CAST CONTEXT:
{roster_context}

CAST GLOBAL BANS (avoid author-voice convergence):
{cast_voice_matrix.get("global_banned_habits", [])}

VOICE CONSTRAINTS FOR THIS CHARACTER (enforce these):
{voice_for_this}

TARGET CHARACTER:
- name: {name}
- slug: {character['slug']}
- role: {role}
- brief: {brief}
"""

    t_diff = Task(
        name="differentiation",
        description=base_context + "\n\n" + DIFFERENTIATOR_PROMPT,
        expected_output="JSON",
        agent=differentiator,
    )

    t_outer = Task(
        name="outer_layer",
        description=base_context + "\n\nUse the differentiation signature from prior step as a constraint.\n\n" + OUTER_LAYER_PROMPT,
        expected_output="JSON",
        agent=outer,
        context=[t_diff],
    )

    t_flesh = Task(
        name="flesh",
        description=base_context + "\n\nUse prior outputs as constraints.\n\n" + FLESH_PROMPT,
        expected_output="JSON",
        agent=flesh,
        context=[t_diff, t_outer],
    )

    t_core = Task(
        name="core",
        description=base_context + "\n\nUse prior outputs as constraints.\n\n" + CORE_PROMPT,
        expected_output="JSON",
        agent=core,
        context=[t_diff, t_outer, t_flesh],
    )

    t_devil = Task(
        name="devils_advocate",
        description=base_context
        + "\n\nHere is the assembled draft from specialists. Critique it hard.\n\n"
        + DEVILS_ADVOCATE_PROMPT,
        expected_output="Markdown",
        agent=devil,
        context=[t_diff, t_outer, t_flesh, t_core],
    )

    t_final = Task(
        name="finalizer",
        description=base_context
        + "\n\nMerge EVERYTHING into final YAML.\n\n"
        + FINALIZER_PROMPT,
        expected_output="JSON",
        agent=finalizer,
        context=[t_diff, t_outer, t_flesh, t_core, t_devil],
    )

    return Crew(
        agents=[differentiator, outer, flesh, core, devil, finalizer],
        tasks=[t_diff, t_outer, t_flesh, t_core, t_devil, t_final],
        process=Process.sequential,
        verbose=True,
    )
