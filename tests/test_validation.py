import polars as pl
import pytest
from pandera.errors import SchemaError

from maptask_vlm.validation import (
    validate_dialogues,
    validate_landmarks,
    validate_utterances,
)


def test_validate_utterances_accepts_valid_data():
    df = pl.DataFrame(
        {
            "dialogue_id": ["q1ec1"],
            "turn_id": [1],
            "speaker": ["g"],
            "utterance": ["hello"],
        }
    )
    validated = validate_utterances(df)
    assert validated.height == 1


def test_validate_utterances_rejects_bad_dialogue_id():
    df = pl.DataFrame(
        {
            "dialogue_id": ["not_valid"],
            "turn_id": [1],
            "speaker": ["g"],
            "utterance": ["hello"],
        }
    )
    with pytest.raises(SchemaError):
        validate_utterances(df)


def test_validate_utterances_rejects_bad_speaker():
    df = pl.DataFrame(
        {
            "dialogue_id": ["q1ec1"],
            "turn_id": [1],
            "speaker": ["x"],
            "utterance": ["hello"],
        }
    )
    with pytest.raises(SchemaError):
        validate_utterances(df)


def test_validate_dialogues_accepts_valid_data():
    df = pl.DataFrame(
        {
            "dialogue_id": ["q1ec1"],
            "quad": [1],
            "conversation": [1],
            "eyecontact": [True],
            "familiar": [False],
            "map_id": ["m12"],
            "devscore": [135],
        }
    )
    validated = validate_dialogues(df)
    assert validated.height == 1


def test_validate_landmarks_accepts_valid_data():
    df = pl.DataFrame(
        {
            "map_id": ["m0"],
            "landmark_name": ["parked van"],
            "landmark_id": ["m0_parked_van"],
            "giver_map_appears": [2],
            "follower_map_appears": [1],
            "giver_referent_status": ["ambiguous"],
            "follower_referent_status": ["unique"],
        }
    )
    validated = validate_landmarks(df)
    assert validated.height == 1


def test_validate_landmarks_rejects_bad_status_value():
    df = pl.DataFrame(
        {
            "map_id": ["m0"],
            "landmark_name": ["parked van"],
            "landmark_id": ["m0_parked_van"],
            "giver_map_appears": [2],
            "follower_map_appears": [1],
            "giver_referent_status": ["not_a_real_status"],
            "follower_referent_status": ["unique"],
        }
    )
    with pytest.raises(SchemaError):
        validate_landmarks(df)
