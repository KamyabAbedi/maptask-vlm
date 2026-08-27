"""Extraction of definite descriptions and candidate landmarks.

Rather than a generic "the + noun" pattern (which matches too much
discourse filler in spontaneous dialogue -- "the top", "the sort of"),
we search each dialogue's utterances for mentions of that dialogue's
own map's known landmark names (from landmarks.parquet), using
word-boundary matching to avoid partial-word false positives
(e.g. "start" inside "starting").
"""

import re

import polars as pl


def find_landmark_mentions(
    utterances: pl.DataFrame, dialogues: pl.DataFrame, landmarks: pl.DataFrame
) -> pl.DataFrame:
    """For every dialogue, search its utterances for mentions of its own
    map's landmark names, using whole-word matching.

    Returns one row per (dialogue_id, turn_id, landmark_name) mention,
    with columns: dialogue_id, turn_id, speaker, utterance, landmark_name.
    """
    dialogue_to_map = dict(
        zip(dialogues["dialogue_id"], dialogues["map_id"], strict=True)
    )

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
            pattern = r"\b" + re.escape(landmark_name) + r"\b"
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

    return pl.DataFrame(records)
