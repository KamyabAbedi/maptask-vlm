from maptask_vlm.loaders import parse_observation_line, parse_transcript_lines


def test_parse_observation_line():
    dialogue_id, count = parse_observation_line(
        "Observation: q1ec1; Result size: 77; atts: 2"
    )
    assert dialogue_id == "q1ec1"
    assert count == 77


def test_parse_observation_line_invalid():
    import pytest

    with pytest.raises(ValueError):
        parse_observation_line("this is not a valid observation line")


def test_parse_transcript_lines():
    lines = ["g\tokay \t", "f\tmmhmm \t"]
    turns = parse_transcript_lines(lines)

    assert turns == [
        {"turn_id": 1, "speaker": "g", "utterance": "okay"},
        {"turn_id": 2, "speaker": "f", "utterance": "mmhmm"},
    ]
