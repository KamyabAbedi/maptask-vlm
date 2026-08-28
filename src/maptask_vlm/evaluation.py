"""Evaluation metrics: parsing raw VLM responses, accuracy, confusion
matrices, and significance tests comparing conditions.
"""

import re

_LETTER_RE = re.compile(r"^\s*([ABC])\)?")

_LETTER_TO_STATUS = {
    "A": "unique",
    "B": "missing",
    "C": "ambiguous",
}


def parse_response_letter(raw_response: str | None) -> str | None:
    """Extract the model's chosen category (unique/missing/ambiguous)
    from its raw response text, based on the leading A/B/C letter.
    Returns None if no recognizable letter is found at the start.
    """
    if raw_response is None:
        return None
    match = _LETTER_RE.match(raw_response)
    if not match:
        return None
    return _LETTER_TO_STATUS[match.group(1)]


_YES_NO_RE = re.compile(r"^\s*(Yes|No)\b", re.IGNORECASE)


def parse_yes_no_response(raw_response: str | None) -> str | None:
    """Extract Yes/No from a giver_ambiguity response, normalized to
    'ambiguous' (Yes) or 'unique' (No) to match the referent_status
    vocabulary used elsewhere.
    """
    if raw_response is None:
        return None
    match = _YES_NO_RE.match(raw_response)
    if not match:
        return None
    return "ambiguous" if match.group(1).lower() == "yes" else "unique"


def parse_predicted_status(
    raw_response: str | None, experiment_type: str
) -> str | None:
    """Parse a model's raw response into a predicted status, using the
    correct parser for the experiment type.
    """
    if experiment_type == "follower_grounding":
        return parse_response_letter(raw_response)
    elif experiment_type == "giver_ambiguity":
        return parse_yes_no_response(raw_response)
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type!r}")
