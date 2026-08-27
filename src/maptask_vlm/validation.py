"""Pandera schemas and validation helpers for processed datasets."""

import pandera.polars as pa
import polars as pl

UtterancesSchema = pa.DataFrameSchema(
    {
        "dialogue_id": pa.Column(str, pa.Check.str_matches(r"^q\d+(ec|nc)\d+$")),
        "turn_id": pa.Column(int, pa.Check.gt(0)),
        "speaker": pa.Column(str, pa.Check.isin(["g", "f"])),
        "utterance": pa.Column(str, pa.Check.str_length(min_value=1)),
    },
    strict=True,
    coerce=True,
)


def validate_utterances(df: pl.DataFrame) -> pl.DataFrame:
    """Validate the utterances table against UtterancesSchema.

    Raises a pandera.errors.SchemaError with details if validation fails.
    Returns the (possibly coerced) DataFrame on success.
    """
    return UtterancesSchema.validate(df)


DialoguesSchema = pa.DataFrameSchema(
    {
        "dialogue_id": pa.Column(
            str, pa.Check.str_matches(r"^q\d+(ec|nc)\d+$"), unique=True
        ),
        "quad": pa.Column(int, pa.Check.in_range(1, 8)),
        "conversation": pa.Column(int, pa.Check.in_range(1, 8)),
        "eyecontact": pa.Column(bool),
        "familiar": pa.Column(bool),
        "map_id": pa.Column(str, pa.Check.str_matches(r"^m\d+$")),
        "devscore": pa.Column(int, pa.Check.ge(0)),
    },
    strict=True,
    coerce=True,
)


def validate_dialogues(df: pl.DataFrame) -> pl.DataFrame:
    """Validate the dialogues table against DialoguesSchema."""
    return DialoguesSchema.validate(df)


LandmarksSchema = pa.DataFrameSchema(
    {
        "map_id": pa.Column(str, pa.Check.str_matches(r"^m\d+$")),
        "landmark_name": pa.Column(str, pa.Check.str_length(min_value=1)),
        "landmark_id": pa.Column(str, unique=True),
        "giver_map_appears": pa.Column(int, pa.Check.ge(0)),
        "follower_map_appears": pa.Column(int, pa.Check.ge(0)),
        "giver_referent_status": pa.Column(
            str, pa.Check.isin(["unique", "missing", "ambiguous"])
        ),
        "follower_referent_status": pa.Column(
            str, pa.Check.isin(["unique", "missing", "ambiguous"])
        ),
    },
    strict=True,
    coerce=True,
)


def validate_landmarks(df: pl.DataFrame) -> pl.DataFrame:
    """Validate the landmarks table against LandmarksSchema."""
    return LandmarksSchema.validate(df)
