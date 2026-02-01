from __future__ import annotations

from pathlib import Path
import re
import json
import yaml
from typing import Optional, Tuple, Any

from rich.console import Console

from .utils import (
    read_text,
    read_yaml,
    write_text,
    write_yaml,
    write_json,
    slugify,
)
from .schemas import CharacterReedsyProfile
from .render import render_markdown
from .crew_factory import build_character_crew
from .cast_crew_factory import build_cast_voice_crew

console = Console()

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "inputs"
OUT = ROOT / "out"


# -----------------------------
# Helpers: Debug artifact writer
# -----------------------------
def write_failure_artifacts(char_dir: Path, phase: str, raw: str, cleaned: str, error: str) -> None:
    char_dir.mkdir(parents=True, exist_ok=True)
    write_text(char_dir / f"_{phase}_raw.txt", raw or "")
    write_text(char_dir / f"_{phase}_cleaned.txt", cleaned or "")
    write_text(char_dir / f"_{phase}_error.txt", error or "Unknown error")


def _build_roster_context(characters: list[dict]) -> str:
    lines = []
    for c in characters:
        lines.append(f"- {c['name']} ({c.get('role','')}): {c.get('brief','')}")
    return "\n".join(lines)


# -----------------------------
# Parsers: JSON primary, YAML fallback (for cast matrix during transition)
# -----------------------------
def _extract_json(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()

    fence = re.search(r"```(?:json)?\s*\n(.*?)\n```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1].strip()

    return t


def try_parse_json(text: str) -> Tuple[Optional[dict], str, Optional[str]]:
    cleaned = _extract_json(text)
    if not cleaned.strip():
        return None, cleaned, "Empty output after cleaning"
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return None, cleaned, f"JSON parsed but not an object (got {type(data).__name__})"
        return data, cleaned, None
    except Exception as e:
        return None, cleaned, repr(e)


def _extract_yaml(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip()

    fence = re.search(r"```(?:yaml)?\s*\n(.*?)\n```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    m = re.search(r"(?m)^[A-Za-z0-9_]+\s*:\s*", t)
    if m:
        return t[m.start() :].strip()

    return t


def try_parse_yaml(text: str) -> Tuple[Optional[dict], str, Optional[str]]:
    cleaned = _extract_yaml(text)
    if not cleaned.strip():
        return None, cleaned, "Empty output after cleaning"
    try:
        data = yaml.safe_load(cleaned)
        if not isinstance(data, dict):
            return None, cleaned, f"YAML parsed but not a mapping/dict (got {type(data).__name__})"
        return data, cleaned, None
    except Exception as e:
        return None, cleaned, repr(e)


def try_parse_structured(text: str) -> Tuple[Optional[dict], str, Optional[str], str]:
    """
    Transitional parser:
    - Try JSON first (preferred)
    - Fall back to YAML (useful if cast crew still returns YAML)
    Returns: (data, cleaned, err, fmt) where fmt is "json"|"yaml"
    """
    data, cleaned, err = try_parse_json(text)
    if not err and data is not None:
        return data, cleaned, None, "json"

    ydata, ycleaned, yerr = try_parse_yaml(text)
    if not yerr and ydata is not None:
        return ydata, ycleaned, None, "yaml"

    # Prefer JSON error messaging if both fail
    return None, cleaned, err or yerr or "Unknown parse error", "json"


# -----------------------------
# CrewAI task output extraction (works across versions)
# -----------------------------
def _task_text(task: Any) -> str:
    """
    CrewAI versions vary: task.output might be a string, an object with .raw, etc.
    This tries a few patterns and always returns a string.
    """
    if task is None:
        return ""
    out = getattr(task, "output", None)
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    raw = getattr(out, "raw", None)
    if isinstance(raw, str):
        return raw
    # last resort
    return str(out)


# -----------------------------
# Main run
# -----------------------------
def run() -> None:
    plot_path = INPUTS / "plot.md"
    chars_path = INPUTS / "characters.yml"

    if not plot_path.exists():
        raise SystemExit(f"Missing {plot_path}. Create inputs/plot.md")
    if not chars_path.exists():
        raise SystemExit(f"Missing {chars_path}. Create inputs/characters.yml")

    plot = read_text(plot_path).strip()
    roster = read_yaml(chars_path)
    characters = roster.get("characters", [])
    if not characters:
        raise SystemExit("characters.yml must contain a top-level 'characters:' list")

    # Assign slugs once, up front
    for c in characters:
        c["slug"] = slugify(c["name"])

    roster_context = _build_roster_context(characters)
    OUT.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) CAST VOICE MATRIX STEP
    # -------------------------
    cast_voice_matrix: dict = {"global_banned_habits": [], "characters": {}}

    try:
        console.rule("[bold]Generating cast voice matrix[/bold]")
        cast_crew = build_cast_voice_crew(plot=plot, characters=characters)
        cast_raw = str(cast_crew.kickoff()).strip()

        # Always write raw
        write_text(OUT / "_raw_cast_voice_matrix.txt", cast_raw)

        cast_data, cast_cleaned, cast_err, cast_fmt = try_parse_structured(cast_raw)
        if cast_err:
            write_text(OUT / "_cast_voice_matrix_error.txt", cast_err)
            write_text(OUT / "_cast_voice_matrix_cleaned.txt", cast_cleaned or "")
            console.print(f"[yellow]Cast voice matrix parse failed; continuing without it[/yellow]: {cast_err}")
        else:
            cast_voice_matrix = cast_data or cast_voice_matrix
            # Prefer JSON output file for canon
            write_json(OUT / "cast_voice_matrix.json", cast_voice_matrix)
            # Optional YAML export (generated by code)
            write_yaml(OUT / "cast_voice_matrix.yml", cast_voice_matrix)
            console.print(f"[green]Cast voice matrix parsed as {cast_fmt}[/green]")

    except Exception as e:
        write_text(OUT / "_cast_voice_matrix_exception.txt", repr(e))
        console.print(f"[yellow]Cast voice matrix generation threw; continuing[/yellow]: {e}")

    # -------------------------
    # 2) PER-CHARACTER STEP
    # -------------------------
    generated: list[CharacterReedsyProfile] = []

    for c in characters:
        name = c["name"]
        slug = c["slug"]
        console.rule(f"[bold]Generating: {name}[/bold]")

        char_dir = OUT / slug
        char_dir.mkdir(parents=True, exist_ok=True)

        try:
            crew = build_character_crew(
                plot=plot,
                roster_context=roster_context,
                character=c,
                cast_voice_matrix=cast_voice_matrix,
            )

            result = crew.kickoff()

            # Write per-task raw outputs (audit trail)
            try:
                for task in getattr(crew, "tasks", []) or []:
                    tname = getattr(task, "name", None) or "task"
                    raw = _task_text(task).strip()
                    if raw:
                        write_text(char_dir / f"_raw_{tname}.txt", raw)
            except Exception as e:
                write_text(char_dir / "_task_dump_exception.txt", repr(e))

            # Final output: parse JSON first; fall back to YAML just in case
            raw_final = str(result).strip()
            write_text(char_dir / "_raw_finalizer_output.txt", raw_final)

            final_data, final_cleaned, final_err, final_fmt = try_parse_structured(raw_final)
            if final_err:
                write_failure_artifacts(char_dir, "finalizer", raw_final, final_cleaned or "", final_err)
                console.print(f"[red]Final structured parse failed for {name}[/red]: {final_err}")
                continue

            # Ensure required fields and attach canon facts from characters.yml
            basics = final_data.get("basics") or {}
            if not basics.get("name"):
                basics["name"] = name
            final_data["basics"] = basics

            final_data["slug"] = slug
            final_data["role"] = c.get("role")
            final_data["brief"] = c.get("brief")
            final_data["gender"] = c.get("gender")
            final_data["orientation"] = c.get("orientation")

            # Attach cast-level voice constraints (if present)
            final_data["voice_constraints"] = cast_voice_matrix.get("characters", {}).get(slug, {})

            # Validate + write canonical JSON + readable MD
            profile = CharacterReedsyProfile.model_validate(final_data)
            generated.append(profile)

            write_json(char_dir / "character.json", profile.model_dump())
            write_text(char_dir / "character.md", render_markdown(profile))

            # Optional YAML export generated by code (uncomment if you want it)
            # write_yaml(char_dir / "character.yml", profile.model_dump())

            console.print(f"[green]Wrote {slug} (parsed as {final_fmt})[/green]")

        except Exception as e:
            write_text(char_dir / "_exception.txt", repr(e))
            console.print(f"[red]Failed for {name}[/red]: {e}")
            continue

    # -------------------------
    # 3) CAST-LEVEL REPORTS
    # -------------------------
    if generated:
        report = _similarity_report(generated)
        write_text(OUT / "_similarity_report.md", report)

    console.rule("[green]Done[/green]")
    console.print(f"Wrote outputs to: {OUT}")


def _similarity_report(profiles: list[CharacterReedsyProfile]) -> str:
    """
    Not ML. Just a pragmatic check:
    - repeated differentiation bullets across multiple characters
    - repeated pet peeves / favorite quote patterns
    - too many identical "wants" or "flaws"
    """
    from collections import Counter

    def norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    diff_bullets = []
    wants = []
    flaws = []
    strengths = []
    tics = []

    for p in profiles:
        # differentiation_signature may be missing depending on schema; guard it
        sig = getattr(p, "differentiation_signature", None) or []
        diff_bullets.extend([norm(x) for x in sig if x])

        # Be defensive: your schema may have nested optional sections
        try:
            wants.append((p.basics.name, norm(getattr(p.psychology, "want_the_most", "") or "")))
            flaws.append((p.basics.name, norm(getattr(p.psychology, "biggest_flaw", "") or "")))
            strengths.append((p.basics.name, norm(getattr(p.psychology, "biggest_strength", "") or "")))
        except Exception:
            pass

        try:
            tics.append((p.basics.name, norm(getattr(p.physical_appearance, "tics_and_mannerisms", "") or "")))
        except Exception:
            pass

    dup_diffs = [k for k, v in Counter(diff_bullets).items() if k and v >= 2]

    def top_dupes(pairs: list[tuple[str, str]], label: str) -> list[str]:
        c = Counter([b for _, b in pairs if b])
        dups = [k for k, v in c.items() if v >= 2]
        lines: list[str] = []
        if dups:
            lines.append(f"### Duplicate {label}")
            for d in dups[:15]:
                names = [n for n, b in pairs if b == d]
                lines.append(f"- **{d}** → {', '.join(names)}")
        return lines

    lines: list[str] = [
        "# Cast Similarity Report",
        "",
        "This is a heuristic scan for clone-risk patterns across the generated bibles.",
        "",
    ]

    if dup_diffs:
        lines += [
            "## Reused Differentiation Bullets (bad sign)",
            *[f"- {d}" for d in dup_diffs[:25]],
            "",
        ]
    else:
        lines += ["## Reused Differentiation Bullets", "- None detected (good).", ""]

    lines += top_dupes(wants, "Wants")
    if lines and lines[-1] != "":
        lines.append("")
    lines += top_dupes(flaws, "Flaws")
    if lines and lines[-1] != "":
        lines.append("")
    lines += top_dupes(strengths, "Strengths")
    if lines and lines[-1] != "":
        lines.append("")
    lines += top_dupes(tics, "Tics/Mannerisms")
    if lines and lines[-1] != "":
        lines.append("")

    lines += [
        "## What to do if you see duplicates",
        "- Change one worldview assumption per duplicate cluster.",
        "- Give each character a different social mask (charming, blunt, evasive, performative, etc.).",
        "- Make their moral-compromise trigger different (money, loyalty, fear, pride, shame).",
        "- Adjust voice constraints: sentence length, metaphor domain, taboo topics, swearing style.",
        "",
    ]
    return "\n".join(lines)
