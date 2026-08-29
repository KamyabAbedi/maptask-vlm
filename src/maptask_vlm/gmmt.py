"""Loader for the GMMT (Grounded Misunderstandings in MapTask) dataset.

GMMT (Li, Gatt & Poesio, LREC 2026) provides perspectivist annotations
for reference expressions in the HCRC Map Task corpus: for each RE, the
giver's intended landmark, the follower's interpreted landmark, and an
understanding_state (aligned / pending / misunderstood) derived from a
5-attribute grounding cascade.

Source: https://github.com/chnln/grounded-misunderstandings-in-maptask
"""

import json
from pathlib import Path

GMMT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "gmmt"
    / "grounded-misunderstandings-in-maptask-main"
)
GMMT_DIALOGUES_DIR = GMMT_DIR / "annotations" / "dialogues"


def parse_gmmt_dialogue_file(path: Path) -> list[dict]:
    """Parse one GMMT per-dialogue JSON file into a flat list of
    reference-expression records.
    """
    data = json.loads(path.read_text())
    dialogue_id = data["dialogue_id"]

    records = []
    for re in data["landmark_reference_expressions"]:
        records.append(
            {
                "ref_id": re["ref_id_unif"],
                "dialogue_id": dialogue_id,
                "concept_id": re["info"]["concept_id"],
                "expression": re["info"]["expression"],
                "speaker": re["info"]["speaker"],
                "addressee": re["info"]["addressee"],
                "utt_id": re["info"]["utt_id"],
                "is_quantificational": re["is_quantificational"],
                "is_specified": re["is_specified"],
                "is_accommodated": re["interpretations"]["is_accommodated"],
                "is_grounded": re["interpretations"]["is_grounded"],
                "is_imagined": re["interpretations"]["is_imagined"],
                "giver_landmark_id": re["interpretations"]["giver"],
                "follower_landmark_id": re["interpretations"]["follower"],
                "understanding_state": re["extra"]["status"],
                "subtype": re["extra"]["subtype"],
                "reason": re["reason"],
            }
        )
    return records


def load_all_gmmt_reference_expressions() -> list[dict]:
    """Parse every GMMT per-dialogue file into one flat list of
    reference-expression records.
    """
    all_records: list[dict] = []
    for path in sorted(GMMT_DIALOGUES_DIR.glob("*.annotated.json")):
        all_records.extend(parse_gmmt_dialogue_file(path))
    return all_records
