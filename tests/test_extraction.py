import polars as pl

from maptask_vlm.extraction import find_landmark_mentions


def test_find_landmark_mentions_basic():
    utterances = pl.DataFrame(
        {
            "dialogue_id": ["q1ec1", "q1ec1"],
            "turn_id": [1, 2],
            "speaker": ["g", "g"],
            "utterance": [
                "go past the abandoned cottage",
                "starting off we head south",
            ],
        }
    )
    dialogues = pl.DataFrame({"dialogue_id": ["q1ec1"], "map_id": ["m12"]})
    landmarks = pl.DataFrame(
        {
            "map_id": ["m12", "m12"],
            "landmark_name": ["abandoned cottage", "start"],
        }
    )

    mentions = find_landmark_mentions(utterances, dialogues, landmarks)

    assert mentions.height == 1
    assert mentions["landmark_name"][0] == "abandoned cottage"


def test_find_landmark_mentions_word_boundary():
    """'start' should not match inside 'starting'."""
    utterances = pl.DataFrame(
        {
            "dialogue_id": ["q1ec1"],
            "turn_id": [1],
            "speaker": ["g"],
            "utterance": ["starting off we head south"],
        }
    )
    dialogues = pl.DataFrame({"dialogue_id": ["q1ec1"], "map_id": ["m12"]})
    landmarks = pl.DataFrame({"map_id": ["m12"], "landmark_name": ["start"]})

    mentions = find_landmark_mentions(utterances, dialogues, landmarks)

    assert mentions.height == 0


def test_find_landmark_mentions_catches_lexical_variant():
    """'mill wheel' should be caught as a mention of 'old mill' on map
    m12, per GMMT's documented lexical variant pairing.
    """
    utterances = pl.DataFrame(
        {
            "dialogue_id": ["q1ec1"],
            "turn_id": [1],
            "speaker": ["f"],
            "utterance": ["i've only got a mill wheel here"],
        }
    )
    dialogues = pl.DataFrame({"dialogue_id": ["q1ec1"], "map_id": ["m12"]})
    landmarks = pl.DataFrame({"map_id": ["m12"], "landmark_name": ["old mill"]})

    mentions = find_landmark_mentions(utterances, dialogues, landmarks)

    assert mentions.height == 1
    assert mentions["landmark_name"][0] == "old mill"
