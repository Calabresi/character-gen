from __future__ import annotations

from pathlib import Path
import re
import json
import yaml
from typing import Optional, Tuple, Any

from rich.console import Console

from .llm import get_llm_config
from .utils import (
    read_text,
    read_yaml,
    write_text,
    write_yaml,
    write_json,
    slugify,
)
from pydantic import ValidationError
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
# Coercion: fix shape mismatches before Pydantic sees them
# -----------------------------

# Role keywords that map a family-list item back into FamilySection's named slots.
_FAMILY_ROLE_KEYWORDS: dict[str, list[str]] = {
    "father": ["father", "dad", "papa", "paternal"],
    "mother": ["mother", "mom", "mama", "maternal"],
}


def _coerce_family(raw: list) -> dict:
    """
    The model likes to return family as a list of relationship objects, e.g.:
        [{"name": "John", "role": "father", "relationship": "..."}]
    FamilySection wants a dict with keyed slots (father, mother, siblings, ...).
    This maps each list item into the right slot by sniffing for role keywords,
    and collapses anything that doesn't match a named slot into 'siblings'.
    """
    result: dict = {}
    leftover: list[str] = []

    for item in raw:
        if not isinstance(item, dict):
            leftover.append(str(item))
            continue

        # Normalise: pull the most common key the model uses for the role label
        item_role = str(
            item.get("role") or item.get("relationship_type") or item.get("type") or ""
        ).lower()
        item_name = item.get("name") or ""

        # Build a FamilyMember-shaped dict from whatever keys the model chose
        member: dict = {
            "relationship_description": (
                item.get("relationship_description")
                or item.get("relationship")
                or item.get("description")
                or item.get("details")
                or f"{item_name} — no description provided"
            ),
        }
        if item.get("age") or item.get("age_if_living"):
            member["age_if_living"] = str(item.get("age") or item.get("age_if_living"))
        if item.get("occupation") or item.get("job"):
            member["occupation"] = str(item.get("occupation") or item.get("job"))

        # Try to land it in a named slot
        placed = False
        for slot, keywords in _FAMILY_ROLE_KEYWORDS.items():
            if any(kw in item_role for kw in keywords) or any(kw in item_name.lower() for kw in keywords):
                result[slot] = member
                placed = True
                break

        if not placed:
            # Doesn't match father/mother — summarise into the siblings catch-all
            desc = member["relationship_description"]
            if item_name:
                desc = f"{item_name}: {desc}"
            leftover.append(desc)

    if leftover:
        result["siblings"] = "; ".join(leftover)

    return result


def _coerce_external_relationships(raw: list) -> dict:
    """
    Same pattern: model returns a list of people, schema wants a flat dict of
    prose fields.  We collapse the list into the closest matching slots.
    """
    result: dict = {}
    friends: list[str] = []
    enemies: list[str] = []
    others: list[str] = []

    for item in raw:
        if not isinstance(item, dict):
            others.append(str(item))
            continue

        item_role = str(
            item.get("role") or item.get("relationship_type") or item.get("type") or ""
        ).lower()
        item_name = item.get("name") or "Unknown"
        desc = (
            item.get("relationship_description")
            or item.get("relationship")
            or item.get("description")
            or item.get("details")
            or ""
        )
        summary = f"{item_name}: {desc}" if desc else item_name

        if any(kw in item_role for kw in ("enemy", "rival", "antagonist", "nemesis")):
            enemies.append(summary)
        elif any(kw in item_role for kw in ("close", "best", "closest", "friend")):
            friends.append(summary)
        else:
            others.append(summary)

    if friends:
        result["closest_friends"] = "; ".join(friends)
    if others:
        result["other_significant_friends"] = "; ".join(others)
    if enemies:
        result["enemies"] = "; ".join(enemies)

    return result


def _coerce_profile_sections(data: dict) -> dict:
    """
    Normalise known sections that the model frequently returns as lists instead
    of dicts.  Safe to call unconditionally — it only touches a field if it's
    actually a list; dicts pass straight through untouched.
    """
    if isinstance(data.get("family"), list):
        data["family"] = _coerce_family(data["family"])

    if isinstance(data.get("external_relationships"), list):
        data["external_relationships"] = _coerce_external_relationships(data["external_relationships"])

    return data



# -----------------------------
# Lenient validation: strip bad fields, never kill a whole character
# -----------------------------

def _validate_leniently(data: dict) -> CharacterReedsyProfile:
    """
    Try strict validation first.  If it fails, read Pydantic's own error
    report to find exactly which top-level keys are broken, move them into
    notes.dropped_fields so nothing is truly lost, and re-validate.  Repeat
    up to 3 times (handles the rare case where stripping one field exposes
    another).  Raises on the final attempt only if we still can't validate —
    at that point something is fundamentally wrong.
    """
    notes: dict = data.get("notes") or {}
    dropped: dict = notes.get("dropped_fields", {})
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        try:
            profile = CharacterReedsyProfile.model_validate(data)
            # Preserve any accumulated dropped-field records
            if dropped:
                profile.notes = profile.notes or {}
                profile.notes["dropped_fields"] = dropped
            return profile
        except ValidationError as exc:
            attempts += 1
            # Extract the set of top-level field names that caused errors.
            # Each error's loc tuple starts with the top-level key.
            bad_keys: set[str] = set()
            for err in exc.errors():
                loc = err.get("loc", ())
                if loc:
                    bad_keys.add(str(loc[0]))

            if not bad_keys:
                # No location info at all — can't surgically fix this
                raise

            # Move each bad key out of data and into the dropped record
            for key in bad_keys:
                if key in data and key not in ("basics", "slug"):
                    # basics and slug are required identity fields — if those
                    # are broken we genuinely can't produce a valid profile
                    dropped[key] = data.pop(key)

            if not dropped:
                # Nothing was actually removable (all errors are in required
                # fields) — stop trying, let it raise
                raise

            data["notes"] = notes
            notes["dropped_fields"] = dropped
            # loop back and try again with the cleaned data

    # Final attempt — let it raise naturally if it still fails
    return CharacterReedsyProfile.model_validate(data)


# -----------------------------
# Stress-test snippet generator
# -----------------------------

# The Devil's Advocate output contains "stress test" scenario prompts.
# We extract them here and use them to generate short proof-of-life
# dialogue/action snippets via a single Ollama call — no crew needed.

_STRESS_TEST_HEADER_RE = re.compile(
    r"(?:stress\s*tests?|scenario\s*prompts?)", re.IGNORECASE
)


def _extract_stress_tests(devil_text: str) -> list[str]:
    """
    Pull the 2–3 stress-test scenario prompts out of the Devil's Advocate
    Markdown output.  They live after a header like '## Stress Tests' or
    '### Stress Test Scenarios', one per bullet.
    """
    if not devil_text:
        return []

    lines = devil_text.splitlines()
    in_section = False
    tests: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect the section header
        if _STRESS_TEST_HEADER_RE.search(stripped) and stripped.startswith("#"):
            in_section = True
            continue
        # Once we're in the section, another header means we've left it
        if in_section and stripped.startswith("#"):
            break
        # Collect bullet lines
        if in_section and stripped.startswith("-"):
            tests.append(stripped.lstrip("- ").strip())

    return tests


def _generate_stress_snippets(
    character: dict,
    voice_constraints: dict,
    stress_tests: list[str],
    llm_cfg: dict,
) -> list[dict]:
    """
    For each stress-test scenario, call Ollama once and ask for a 3–5 line
    dialogue or action snippet that proves the character's voice is distinct.
    Returns a list of {scenario, snippet} dicts.  Failures on individual
    scenarios are silently skipped — this is a best-effort enrichment pass.
    """
    from crewai.llm import LLM  # local import — only needed when this runs

    results: list[dict] = []
    llm = LLM(model=llm_cfg["model"], base_url=llm_cfg["base_url"])

    name = character.get("name", "the character")
    voice_summary = json.dumps(voice_constraints, ensure_ascii=False) if voice_constraints else "none"

    for scenario in stress_tests:
        prompt = (
            f"Character: {name}\n"
            f"Voice constraints: {voice_summary}\n\n"
            f"Scenario: {scenario}\n\n"
            f"Write 3–5 lines of dialogue or internal action showing how this "
            f"character responds in this exact scenario.  Make their voice "
            f"unmistakable.  No narration wrapper, no explanation — just the snippet.\n"
        )
        try:
            response = llm.call([{"role": "user", "content": prompt}])
            snippet = str(response).strip() if response else ""
            if snippet:
                results.append({"scenario": scenario, "snippet": snippet})
        except Exception:
            continue  # skip this scenario, keep going

    return results


# -----------------------------
# Fallback assembler: merge specialist outputs when the finalizer fails
# -----------------------------

# Maps task name -> which top-level keys its output is expected to contribute.
# We only merge keys we actually expect; anything else in a task output is ignored.
_TASK_CONTRIBUTES: dict[str, list[str]] = {
    "differentiation":       ["not_the_author_because", "worldview_assumptions", "voice_constraints", "contradictions"],
    "outer_layer":           ["basics", "physical_appearance", "speech_and_communication"],
    "flesh":                 ["past", "family", "external_relationships"],
    "core":                  ["psychology", "present_and_future"],
    "devils_advocate":       [],  # Markdown, not JSON — handled separately below
}


def _assemble_from_tasks(crew: Any, character: dict) -> dict | None:
    """
    Parse each specialist task's output individually and merge into one dict.
    Returns None only if zero tasks produced usable JSON — i.e. truly nothing
    to work with.  A partial profile (even just basics + one section) is still
    better than dropping the character entirely.
    """
    merged: dict = {}
    parsed_any = False

    for task in getattr(crew, "tasks", []) or []:
        tname = getattr(task, "name", None) or ""
        raw = _task_text(task).strip()
        if not raw:
            continue

        # Devil's advocate is Markdown, not JSON.  Extract bullet text and
        # stash it as clone_risks_and_fixes so it survives into the final profile.
        if tname == "devils_advocate":
            bullets = [
                line.lstrip("- ").strip()
                for line in raw.splitlines()
                if line.strip().startswith("-")
            ]
            if bullets:
                merged["clone_risks_and_fixes"] = bullets
                parsed_any = True
            continue

        # For every other task, try to parse JSON out of it
        data, _, err = try_parse_json(raw)
        if err or not data:
            continue

        parsed_any = True

        # If this task produced a differentiation signature, flatten it into
        # the single list the top-level schema expects
        if tname == "differentiation":
            sig_parts: list[str] = []
            for key in ("not_the_author_because", "worldview_assumptions", "contradictions"):
                val = data.get(key)
                if isinstance(val, list):
                    sig_parts.extend(str(v) for v in val if v)
            if sig_parts:
                merged["differentiation_signature"] = sig_parts
            # voice_constraints from the differentiator is per-character notes,
            # not the cast matrix — skip it here, it gets stamped later from
            # cast_voice_matrix.
            continue

        # For all other specialist tasks, just merge in the keys we expect
        expected_keys = _TASK_CONTRIBUTES.get(tname, [])
        for key in expected_keys:
            if key in data and key not in merged:
                merged[key] = data[key]

    if not parsed_any:
        return None

    return merged


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
            devil_raw = ""
            try:
                for task in getattr(crew, "tasks", []) or []:
                    tname = getattr(task, "name", None) or "task"
                    raw = _task_text(task).strip()
                    if raw:
                        write_text(char_dir / f"_raw_{tname}.txt", raw)
                    if tname == "devils_advocate":
                        devil_raw = raw
            except Exception as e:
                write_text(char_dir / "_task_dump_exception.txt", repr(e))

            # ----------------------------------------------------------
            # Assembly: try finalizer output first; fall back to
            # per-task merge if it doesn't parse.  Either way we end up
            # with a single dict that we then stamp and validate.
            # ----------------------------------------------------------
            raw_final = str(result).strip()
            write_text(char_dir / "_raw_finalizer_output.txt", raw_final)

            final_data, final_cleaned, final_err, final_fmt = try_parse_structured(raw_final)

            if final_err or not final_data:
                # Finalizer choked — assemble from the specialist tasks instead
                write_failure_artifacts(char_dir, "finalizer", raw_final, final_cleaned or "", final_err or "")
                console.print(f"[yellow]Finalizer failed for {name}, assembling from specialist outputs[/yellow]")

                final_data = _assemble_from_tasks(crew, c)
                final_fmt = "assembled"

                if not final_data:
                    console.print(f"[red]Zero usable task outputs for {name} — skipping[/red]")
                    continue

            # Coerce any list-shaped sections into the dicts the schema expects
            final_data = _coerce_profile_sections(final_data)

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
            char_voice = cast_voice_matrix.get("characters", {}).get(slug, {})
            final_data["voice_constraints"] = char_voice

            # ----------------------------------------------------------
            # Stress-test snippets: extract scenarios from Devil's
            # Advocate, generate proof-of-life snippets, stamp into notes
            # ----------------------------------------------------------
            stress_tests = _extract_stress_tests(devil_raw)
            if stress_tests:
                from .llm import get_llm_config as _get_llm  # already imported at module level but be explicit
                snippets = _generate_stress_snippets(c, char_voice, stress_tests, get_llm_config())
                if snippets:
                    notes = final_data.get("notes") or {}
                    notes["stress_test_snippets"] = snippets
                    final_data["notes"] = notes

            # ----------------------------------------------------------
            # Validate leniently — strips bad fields into notes rather
            # than killing the character, then write outputs
            # ----------------------------------------------------------
            profile = _validate_leniently(final_data)
            generated.append(profile)

            write_json(char_dir / "character.json", profile.model_dump())
            write_text(char_dir / "character.md", render_markdown(profile))

            console.print(f"[green]Wrote {slug} (source: {final_fmt})[/green]")

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
    Two-layer similarity scan:

    Layer 1 — Exact match (original behaviour): catches literal duplicate
    bullets across differentiation_signature, wants, flaws, strengths, tics.

    Layer 2 — Semantic drift (new): for every pair of characters, computes
    TF-IDF cosine similarity over the concatenation of their psychology fields.
    Pairs above a configurable threshold get flagged as "drifting toward each
    other" even if they share zero literal phrases.  Falls back gracefully to
    Layer 1 only if sklearn is unavailable.
    """
    from collections import Counter

    SEMANTIC_THRESHOLD = 0.45  # pairs above this are flagged

    def norm(s: str) -> str:
        return " ".join((s or "").lower().split())

    # ----------------------------------------------------------
    # Layer 1: exact-match duplicate detection (unchanged logic)
    # ----------------------------------------------------------
    diff_bullets: list[str] = []
    wants: list[tuple[str, str]] = []
    flaws: list[tuple[str, str]] = []
    strengths: list[tuple[str, str]] = []
    tics: list[tuple[str, str]] = []

    for p in profiles:
        sig = getattr(p, "differentiation_signature", None) or []
        diff_bullets.extend([norm(x) for x in sig if x])

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

    # ----------------------------------------------------------
    # Layer 2: semantic drift via TF-IDF cosine similarity
    # ----------------------------------------------------------
    # Fields we concatenate per character for the semantic comparison.
    _PSYCH_FIELDS = [
        "want_the_most", "biggest_flaw", "biggest_strength", "biggest_fear",
        "secrets_they_keep", "moral_compass_and_compromise", "perfect_happiness",
        "remembered_for", "approach_to_power", "approach_to_ambition",
        "approach_to_love", "approach_to_change",
    ]

    def _psych_text(p: CharacterReedsyProfile) -> str:
        parts: list[str] = []
        psych = getattr(p, "psychology", None)
        if not psych:
            return ""
        for field in _PSYCH_FIELDS:
            val = getattr(psych, field, None)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    drift_pairs: list[tuple[str, str, float]] = []  # (name_a, name_b, score)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        import numpy as np

        texts = [_psych_text(p) for p in profiles]
        names = [p.basics.name for p in profiles]

        # Only run if we have at least 2 characters with actual text
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if len(non_empty) >= 2:
            indices, corpus = zip(*non_empty)
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(list(corpus))
            similarity_matrix = cos_sim(tfidf_matrix)

            for row_idx in range(len(indices)):
                for col_idx in range(row_idx + 1, len(indices)):
                    score = float(similarity_matrix[row_idx, col_idx])
                    if score >= SEMANTIC_THRESHOLD:
                        drift_pairs.append((
                            names[indices[row_idx]],
                            names[indices[col_idx]],
                            round(score, 3),
                        ))
            # Sort by similarity descending — worst offenders first
            drift_pairs.sort(key=lambda x: x[2], reverse=True)

    except ImportError:
        pass  # sklearn not available — Layer 2 just doesn't run, no crash

    # ----------------------------------------------------------
    # Assemble the report
    # ----------------------------------------------------------
    lines: list[str] = [
        "# Cast Similarity Report",
        "",
        "This is a heuristic scan for clone-risk patterns across the generated bibles.",
        "",
    ]

    # Layer 1 output
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

    # Layer 2 output
    if drift_pairs:
        lines += [
            "## Semantic Drift Warnings (psychology fields drifting toward each other)",
            "",
            "These character pairs share no literal phrases but their core psychology",
            "fields score above the similarity threshold. They risk reading as the same",
            "person once prose smooths out the surface differences.",
            "",
        ]
        for name_a, name_b, score in drift_pairs:
            lines.append(f"- **{name_a}** ↔ **{name_b}** — similarity: {score}")
        lines.append("")
    else:
        lines += [
            "## Semantic Drift Warnings",
            "- No psychology drift detected (good).",
            "",
        ]

    lines += [
        "## What to do if you see duplicates",
        "- Change one worldview assumption per duplicate cluster.",
        "- Give each character a different social mask (charming, blunt, evasive, performative, etc.).",
        "- Make their moral-compromise trigger different (money, loyalty, fear, pride, shame).",
        "- Adjust voice constraints: sentence length, metaphor domain, taboo topics, swearing style.",
        "",
    ]
    return "\n".join(lines)
