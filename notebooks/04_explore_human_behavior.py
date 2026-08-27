import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, mo, pl


@app.cell
def _(pl):
    mentions = pl.read_parquet("data/interim/definite_descriptions.parquet")
    utterances = pl.read_parquet("data/processed/utterances.parquet")

    mentions.shape, utterances.shape
    return mentions, utterances


@app.cell
def _(mentions, pl):
    ambiguous_sample = mentions.filter(
        pl.col("giver_referent_status") == "ambiguous"
    ).head(5)

    ambiguous_sample
    return


@app.cell
def _(pl, utterances):
    q1ec3_utterances = utterances.filter(pl.col("dialogue_id") == "q1ec3").sort("turn_id")

    # Context around turn 11 (first mention)
    q1ec3_utterances.filter(pl.col("turn_id").is_between(8, 18))
    return


@app.cell
def _(pl, utterances):
    utterances.filter(pl.col("dialogue_id") == "q1ec3").filter(pl.col("turn_id") == 11)["utterance"][0]
    return


@app.cell
def _(mentions, pl):
    mentions.filter(pl.col("dialogue_id") == "q1ec3").filter(pl.col("turn_id") == 11)
    return


@app.cell
def _(mentions, pl, utterances):
    def add_next_turn_context(mentions: pl.DataFrame, utterances: pl.DataFrame) -> pl.DataFrame:
        """Attach the next turn's utterance (same dialogue) to each mention,
        to inspect the listener's immediate reaction.
        """
        next_turn_lookup = utterances.select(
            "dialogue_id",
            "turn_id",
            pl.col("speaker").alias("next_speaker"),
            pl.col("utterance").alias("next_utterance"),
        ).with_columns(
            (pl.col("turn_id") - 1).alias("turn_id")  # shift so it joins to the PRIOR turn
        )

        return mentions.join(next_turn_lookup, on=["dialogue_id", "turn_id"], how="left")


    mentions_with_context = add_next_turn_context(mentions, utterances)
    mentions_with_context.filter(
        (pl.col("dialogue_id") == "q1ec3") & (pl.col("turn_id") == 11)
    )
    return (mentions_with_context,)


@app.cell
def _(mentions_with_context, pl):
    missing_cases = mentions_with_context.filter(
        pl.col("follower_referent_status") == "missing"
    )

    # Simple keyword-based detector for explicit "I don't have this" style repairs
    repair_keywords = ["don't have", "no ", "hang on", "haven't got", "not got"]

    missing_cases_with_flag = missing_cases.with_columns(
        pl.col("next_utterance")
        .str.to_lowercase()
        .str.contains("|".join(repair_keywords))
        .alias("looks_like_explicit_repair")
    )

    missing_cases_with_flag["looks_like_explicit_repair"].value_counts()
    return (missing_cases_with_flag,)


@app.cell
def _(missing_cases_with_flag, pl):
    missing_cases_with_flag.filter(pl.col("looks_like_explicit_repair").is_null())
    return


@app.cell
def _(missing_cases_with_flag, pl):
    missing_cases_with_flag.filter(pl.col("looks_like_explicit_repair") == True).select(
        "dialogue_id", "turn_id", "utterance", "landmark_name", "next_utterance"
    ).head(8)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Week 6 findings: human repair behavior around missing referents

    Of 1,080 mentions where the giver refers to a landmark missing from
    the follower's map, ~9% (95) show an immediate, explicit verbal
    repair from the follower (e.g. "i don't have a slate mountain",
    "no no graveyard").

    Real examples show a range of repair sophistication:
    - Simple negation: "don't have a slate mountain"
    - Repeated confusion: "graveyard" was flagged as missing twice in
      the same dialogue (q1ec3, turns 11 and 42), suggesting the giver
      didn't fully register the first correction.
    - Active meaning negotiation: "i don't have an old mill it says
      mill wheel must be the same thing" -- the follower doesn't just
      reject the reference, they propose a mapping to a landmark they
      do have.

    The 91% majority without an explicit repair keyword likely include:
    quieter acknowledgments (a plain "mmhmm" while confused), repairs
    phrased without our keyword list's exact words, or cases where the
    giver moves on before the follower can react. A more thorough
    analysis (e.g. reading a larger random sample, or using dialogue-act
    annotations already present in the corpus's `moves/` directory --
    unexplored so far) would refine this estimate.
    """)
    return


@app.cell
def _(Path, mentions_with_context):
    context_output_path = Path("data/interim/mentions_with_context.parquet")
    mentions_with_context.write_parquet(context_output_path)

    context_output_path, mentions_with_context.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
