"""Loaders for the HCRC Map Task corpus.

This module is responsible for reading files out of ``data/raw/`` and turning
them into Python objects. Nothing here should ever write back into
``data/raw/`` -- that directory is treated as read-only, immutable source
data.
"""

import re
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TRANSCRIPTS_DIR = RAW_DATA_DIR / "hcrc_maptask" / "Transcripts"

_OBSERVATION_RE = re.compile(r"Observation:\s*(\S+);\s*Result size:\s*(\d+)")


def list_raw_files(subdir: str = "") -> list[Path]:
    """List files under data/raw/<subdir>, for quick inspection in notebooks."""
    target = RAW_DATA_DIR / subdir
    if not target.exists():
        return []
    return sorted(p for p in target.rglob("*") if p.is_file())


def parse_observation_line(line: str) -> tuple[str, int]:
    """Extract (dialogue_id, expected_row_count) from a transcript's
    'Observation: <id>; Result size: <n>; atts: 2' metadata line.
    """
    match = _OBSERVATION_RE.match(line)
    if not match:
        raise ValueError(f"Unexpected Observation line format: {line!r}")
    return match.group(1), int(match.group(2))


def parse_transcript_lines(lines: list[str]) -> list[dict]:
    """Parse transcript data lines (after metadata + header) into turn
    records with 1-indexed turn_id, speaker, and utterance.
    """
    records = []
    for turn_id, line in enumerate(lines, start=1):
        parts = line.split("\t")
        records.append(
            {
                "turn_id": turn_id,
                "speaker": parts[0].strip(),
                "utterance": parts[1].strip(),
            }
        )
    return records


def parse_transcript_file(path: Path) -> list[dict]:
    """Parse one HCRC transcript file into turn records tagged with
    dialogue_id, validated against the file's own declared Result size.
    """
    lines = path.read_text().splitlines()
    dialogue_id, expected_count = parse_observation_line(lines[1])

    data_lines = lines[3:]  # skip "Found 2 args.", Observation line, header
    turns = parse_transcript_lines(data_lines)

    if len(turns) != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} rows, got {len(turns)}"
        )

    for turn in turns:
        turn["dialogue_id"] = dialogue_id

    return turns


def load_all_transcripts() -> list[dict]:
    """Parse every transcript file in the corpus into a flat list of
    turn records.
    """
    all_turns: list[dict] = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
        all_turns.extend(parse_transcript_file(path))
    return all_turns
