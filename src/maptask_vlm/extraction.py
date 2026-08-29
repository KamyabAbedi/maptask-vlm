"""Extraction of definite descriptions and candidate landmarks.

Rather than a generic "the + noun" pattern (which matches too much
discourse filler in spontaneous dialogue -- "the top", "the sort of"),
we search each dialogue's utterances for mentions of that dialogue's
own map's known landmark names (from landmarks.parquet), using
word-boundary matching to avoid partial-word false positives
(e.g. "start" inside "starting").

Also searches for GMMT-documented lexical variants (e.g. "mill wheel"
as the follower-side name for the giver-side "old mill") so mentions
using either name are found.
"""

import json
import re
from pathlib import Path

import polars as pl

GMMT_VARIANTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "gmmt"
    / "grounded-misunderstandings-in-maptask-main"
    / "annotations"
    / "assets"
    / "lexical_variant_landmark_info.json"
)


def load_lexical_variants() -> dict[str, str]:
    """Load GMMT's lexical variant pairs (giver's landmark concept ID ->
    follower's landmark concept ID for the same physical landmark),
    e.g. 'm12_old_mill' -> 'm12_mill_wheel'.
    """
    data = json.loads(GMMT_VARIANTS_PATH.read_text())
    pairs = data["landmarks"][0]
    return {pair["landmark_giver_map"]: pair["landmark_follower_map"] for pair in pairs}


def _concept_id(map_id: str, landmark_name: str) -> str:
    """'m12', 'old mill' -> 'm12_old_mill' (GMMT's concept ID format)."""
    return f"{map_id}_{landmark_name.replace(' ', '_')}"


def _name_from_concept_id(map_id: str, concept_id: str) -> str:
    """'m12', 'm12_mill_wheel' -> 'mill wheel' (strip map prefix, underscores to spaces)."""
    prefix = f"{map_id}_"
    stripped = (
        concept_id.removeprefix(prefix)
    )
    return stripped.replace("_", " ")


def find_landmark_mentions(
    utterances: pl.DataFrame, dialogues: pl.DataFrame, landmarks: pl.DataFrame
) -> pl.DataFrame:
    """For every dialogue, search its utterances for mentions of its own
    map's landmark names (plus any GMMT-documented lexical variant),
    using whole-word matching.

    Returns one row per (dialogue_id, turn_id, landmark_name) mention,
    with columns: dialogue_id, turn_id, speaker, utterance, landmark_name.
    landmark_name is always the canonical (original) name, even when
    the match came from searching a variant spelling.
    """
    dialogue_to_map = dict(
        zip(dialogues["dialogue_id"], dialogues["map_id"], strict=True)
    )

    variants = load_lexical_variants()
    # Also build the reverse direction, so either spelling triggers a match.
    variants_reverse = {v: k for k, v in variants.items()}

    map_to_landmarks: dict[str, list[str]] = {}
    for map_id, name in zip(
        landmarks["map_id"], landmarks["landmark_name"], strict=True
    ):
        map_to_landmarks.setdefault(map_id, []).append(name)

    records = []
    for dialogue_id, turn_id, speaker, utterance in utterances.select(
        "dialogue_id", "turn_id", "speaker", "utterance"
    ).iter_rows():
        map_id = dialogue_to_map.get(dialogue_id)
        if map_id is None:
            continue

        for landmark_name in map_to_landmarks.get(map_id, []):
            concept_id = _concept_id(map_id, landmark_name)

            search_names = {landmark_name}
            if concept_id in variants:
                search_names.add(_name_from_concept_id(map_id, variants[concept_id]))
            if concept_id in variants_reverse:
                search_names.add(
                    _name_from_concept_id(map_id, variants_reverse[concept_id])
                )

            for search_name in search_names:
                pattern = r"\b" + re.escape(search_name) + r"\b"
                if re.search(pattern, utterance):
                    records.append(
                        {
                            "dialogue_id": dialogue_id,
                            "turn_id": turn_id,
                            "speaker": speaker,
                            "utterance": utterance,
                            "landmark_name": landmark_name,
                        }
                    )
                    break  # avoid double-counting if both names somehow match

    return pl.DataFrame(records)
