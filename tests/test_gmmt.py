import polars as pl

from maptask_vlm.gmmt import parse_gmmt_dialogue_file
from maptask_vlm.validation import validate_gmmt_reference_expressions


def test_parse_gmmt_dialogue_file_structure(tmp_path):
    sample = {
        "dialogue_id": "q1ec1",
        "landmark_reference_expressions": [
            {
                "ref_id_unif": "q1ec1.ref.0",
                "is_quantificational": False,
                "is_specified": True,
                "interpretations": {
                    "is_accommodated": True,
                    "is_grounded": True,
                    "is_imagined": False,
                    "giver": "m12_caravan_park@g",
                    "follower": "m12_caravan_park@f",
                },
                "reason": "test reason",
                "extra": {"status": "aligned", "subtype": "grounded"},
                "info": {
                    "concept_id": "m12_caravan_park",
                    "speaker": "giver",
                    "addressee": "follower",
                    "expression": "a caravan park",
                    "utt_id": 1,
                },
            }
        ],
    }
    path = tmp_path / "q1ec1.annotated.json"
    path.write_text(__import__("json").dumps(sample))

    records = parse_gmmt_dialogue_file(path)

    assert len(records) == 1
    assert records[0]["ref_id"] == "q1ec1.ref.0"
    assert records[0]["understanding_state"] == "aligned"
    assert records[0]["giver_landmark_id"] == "m12_caravan_park@g"


def test_validate_gmmt_reference_expressions_accepts_valid_data():
    df = pl.DataFrame(
        {
            "ref_id": ["q1ec1.ref.0"],
            "dialogue_id": ["q1ec1"],
            "concept_id": ["m12_caravan_park"],
            "expression": ["a caravan park"],
            "speaker": ["giver"],
            "addressee": ["follower"],
            "utt_id": [1],
            "understanding_state": ["aligned"],
        }
    )
    validated = validate_gmmt_reference_expressions(df)
    assert validated.height == 1
