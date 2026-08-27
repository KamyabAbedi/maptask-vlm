
from maptask_vlm.transforms import turns_to_dataframe


def test_turns_to_dataframe_schema():
    turns = [
        {"turn_id": 1, "speaker": "g", "utterance": "hello", "dialogue_id": "q1ec1"},
        {"turn_id": 2, "speaker": "f", "utterance": "hi", "dialogue_id": "q1ec1"},
    ]
    df = turns_to_dataframe(turns)

    assert df.columns == ["dialogue_id", "turn_id", "speaker", "utterance"]
    assert df.height == 2
    assert df["dialogue_id"][0] == "q1ec1"
