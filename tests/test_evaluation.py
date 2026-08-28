from maptask_vlm.evaluation import (
    parse_predicted_status,
    parse_response_letter,
    parse_yes_no_response,
)


def test_parse_response_letter_basic():
    assert parse_response_letter("A) I can see it and understand") == "unique"
    assert parse_response_letter("B) I don't see it") == "missing"
    assert parse_response_letter(" C) ask which one") == "ambiguous"


def test_parse_response_letter_no_match():
    assert parse_response_letter("nonsense response") is None
    assert parse_response_letter(None) is None


def test_parse_yes_no_response():
    assert parse_yes_no_response("Yes, it appears twice") == "ambiguous"
    assert parse_yes_no_response("No, it only appears once") == "unique"
    assert parse_yes_no_response("maybe") is None
    assert parse_yes_no_response(None) is None


def test_parse_predicted_status_routes_by_experiment_type():
    assert parse_predicted_status("A) proceed", "follower_grounding") == "unique"
    assert parse_predicted_status("Yes, twice", "giver_ambiguity") == "ambiguous"


def test_parse_predicted_status_unknown_experiment_type_raises():
    import pytest

    with pytest.raises(ValueError):
        parse_predicted_status("A) proceed", "not_a_real_experiment")
