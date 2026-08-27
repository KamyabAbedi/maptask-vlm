from maptask_vlm.prompts import (
    build_follower_grounding_prompt,
    build_giver_ambiguity_prompt,
)


def test_build_follower_grounding_prompt_includes_utterance():
    prompt = build_follower_grounding_prompt("go past the abandoned cottage")
    assert "go past the abandoned cottage" in prompt
    assert "A)" in prompt
    assert "B)" in prompt
    assert "C)" in prompt


def test_build_giver_ambiguity_prompt_includes_landmark():
    prompt = build_giver_ambiguity_prompt("parked van")
    assert "parked van" in prompt
    assert "Yes or No" in prompt
