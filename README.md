# Character Bible Crew (Reedsy-style)

Generates Reedsy-structured character profiles (JSON + Markdown) using CrewAI and local Ollama.

## Setup

1) Install deps (pick one):
- uv: `uv venv && uv pip install -e .`
- pip: `python -m venv .venv && .venv/Scripts/activate && pip install -e .`

2) Copy env:
- `cp .env.example .env`

3) Create inputs:
- `inputs/plot.md`
- `inputs/characters.yml`

4) Run:
- `character-bible`

Outputs:
- `out/<slug>/character.json`
- `out/<slug>/character.md`
- `out/_similarity_report.md`
