"""Prompt templates used to query VLMs about referent grounding.

Two experiment types:
- follower_grounding: model plays the listener, sees the follower's map,
  and must classify how it would respond to the giver's instruction.
- giver_ambiguity: model plays the speaker, sees the giver's map, and
  must judge whether a landmark it's about to reference is ambiguous
  (appears more than once) on that same map.
"""

FOLLOWER_GROUNDING_PROMPT = """You are looking at a hand-drawn map. Someone is guiding you along a route and just said:

"{definite_description_context}"

Based on what you can see on your map, how do you respond?

A) I can see it and understand -- proceed
B) I don't see that landmark on my map
C) I see more than one possible match -- ask which one

Answer with just the letter, then a one-sentence explanation."""


GIVER_AMBIGUITY_PROMPT = """You are looking at a hand-drawn map that you'll use to give someone directions. You're about to refer to "{landmark}" in your instructions.

Is this landmark name ambiguous on your own map (i.e., does it appear more than once)?

Answer Yes or No, then a one-sentence explanation."""


def build_follower_grounding_prompt(definite_description_context: str) -> str:
    """Build the prompt for the follower_grounding experiment."""
    return FOLLOWER_GROUNDING_PROMPT.format(
        definite_description_context=definite_description_context
    )


def build_giver_ambiguity_prompt(landmark: str) -> str:
    """Build the prompt for the giver_ambiguity experiment."""
    return GIVER_AMBIGUITY_PROMPT.format(landmark=landmark)
