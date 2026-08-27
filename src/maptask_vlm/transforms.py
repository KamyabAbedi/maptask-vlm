"""Transforms: raw parsed corpus objects -> tidy Polars tables."""

import polars as pl


def turns_to_dataframe(turns: list[dict]) -> pl.DataFrame:
    """Convert a list of turn record dicts into the canonical
    dialogue_turns schema:

        dialogue_id: str
        turn_id: int
        speaker: str
        utterance: str
    """
    return pl.DataFrame(turns).select("dialogue_id", "turn_id", "speaker", "utterance")
