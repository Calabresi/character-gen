from __future__ import annotations

from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

from .llm import get_llm_config
from .cast_prompts import CAST_SYSTEM, CAST_VOICE_MATRIX_PROMPT
from .utils import slugify

def build_cast_voice_crew(plot: str, characters: list[dict]) -> Crew:
    llm_cfg = get_llm_config()
    llm = LLM(model=llm_cfg["model"], base_url=llm_cfg["base_url"], provider="ollama")

    # Build a stable roster display (with slugs)
    roster_lines = []
    for c in characters:
        roster_lines.append(
            f"- {c['name']} | slug: {slugify(c['name'])} | role: {c.get('role','')} | brief: {c.get('brief','')}"
        )
    roster = "\n".join(roster_lines)

    agent = Agent(
        role="Voice Director",
        goal="Create enforceable, distinct voice constraints per character and global cast bans.",
        backstory="You specialize in dialogue differentiation and anti-author-voice constraints.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        system=CAST_SYSTEM,
    )

    task = Task(
        description=f"""PLOT:
{plot}

CAST:
{roster}

{CAST_VOICE_MATRIX_PROMPT}
""",
        expected_output="YAML",
        agent=agent,
    )

    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
