import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():

    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    mo.md("""
    # Can VLMs Ground Definite Descriptions in Asymmetric Visual Contexts?

    ## Research Question

    Using the HCRC Map Task Corpus, this project asks: can vision-language
    models, playing the role of a listener following spoken route
    instructions, correctly determine whether a referenced landmark is
    **unique** (proceed), **missing** (report it's not there), or
    **ambiguous** (ask for clarification) -- the way human listeners
    naturally do?
    """)
    return


@app.cell
def _(mo, pl):
    dialogues = pl.read_parquet("data/processed/dialogues.parquet")
    utterances = pl.read_parquet("data/processed/utterances.parquet")
    landmarks = pl.read_parquet("data/processed/landmarks.parquet")
    experiment_cases = pl.read_parquet("data/processed/experiment_cases.parquet")

    mo.md(
        f"""
        ## Dataset

        Built from the **HCRC Map Task Corpus** (128 dialogues, freely
        available under CC BY 4.0 from the University of Edinburgh):

        | Table | Rows | Description |
        |---|---|---|
        | `dialogues.parquet` | {dialogues.height} | One row per dialogue: quad, eye-contact condition, assigned map |
        | `utterances.parquet` | {utterances.height} | One row per dialogue turn: speaker, text |
        | `landmarks.parquet` | {landmarks.height} | One row per (map, landmark): giver/follower appearance counts, referent status |
        | `experiment_cases.parquet` | {experiment_cases.height} | Full set of landmark mentions eligible for VLM testing |

        **Key discovery (Week 2):** the corpus's own NXT annotations
        (`maptask-corpus.xml`, `maptask-landmarks.xml`) directly encode
        ground-truth referent status per landmark per map -- no manual
        image inspection or vision-model labeling was needed to determine
        whether a landmark is unique, missing, or ambiguous.
        """
    )
    return


@app.cell
def _(mo, pl):
    mentions_with_context = pl.read_parquet("data/interim/mentions_with_context.parquet")

    missing_mentions = mentions_with_context.filter(pl.col("follower_referent_status") == "missing")
    n_missing = missing_mentions.height
    n_explicit_repair = missing_mentions.filter(
        pl.col("next_utterance").str.to_lowercase().str.contains(
            "don't have|no |hang on|haven't got|not got"
        )
    ).height

    ambiguous_giver_mentions = mentions_with_context.filter(pl.col("giver_referent_status") == "ambiguous").height

    mo.md(
        f"""
        ## Human Baseline (Week 6)

        Analysis of real human dialogue behavior around referent mentions:

        - **{n_missing} mentions** where the giver referenced a landmark
          missing from the follower's map. Of these, **{n_explicit_repair}
          (~{n_explicit_repair/n_missing:.0%})** show an immediate, explicit
          verbal repair (e.g. *"i don't have a slate mountain"*).
        - Repair sophistication varies: simple negation, repeated confusion
          across turns, and active meaning negotiation (e.g. a follower
          proposing "mill wheel" as equivalent to the giver's "old mill").
        - **{ambiguous_giver_mentions} mentions** reference a landmark that
          is ambiguous (appears twice) on the giver's own map -- ambiguity
          in this corpus is exclusively a giver-side phenomenon; follower
          maps never contain duplicate landmarks.
        """
    )
    return


@app.cell
def _(mo, pl):
    scored = pl.read_parquet("results/tables/evaluation_scored.parquet")

    follower_accuracy = (
        scored.filter(pl.col("experiment_type") == "follower_grounding")
        .group_by("model")
        .agg(pl.col("correct").mean().alias("accuracy"))
        .sort("model")
    )

    mo.md(
        f"""
        ## VLM Results: follower_grounding (balanced 225 unique / 225 missing)

        | Model | Accuracy |
        |---|---|
    {chr(10).join(f"    | {row['model']} | {row['accuracy']:.1%} |" for row in follower_accuracy.iter_rows(named=True))}

        **Chance level: 50%.** All three tested open-weight VLMs perform at
        or below chance, driven by strong, largely input-independent
        response biases rather than genuine per-case visual grounding.
        """
    )
    return follower_accuracy, scored


@app.cell
def _(pl, scored):
    bias_breakdown = (
        scored.filter(pl.col("experiment_type") == "follower_grounding")
        .group_by(["model", "predicted_status"])
        .agg(pl.count().alias("n"))
        .with_columns((pl.col("n") / 450 * 100).round(1).alias("pct"))
        .sort(["model", "predicted_status"])
    )

    bias_breakdown
    return (bias_breakdown,)


@app.cell
def _(bias_breakdown, mo):
    pivot = bias_breakdown.pivot(
        on="predicted_status", index="model", values="pct"
    ).sort("model")

    mo.md(
        f"""
        ### Response distribution (% of 450 follower_grounding cases)

        | Model | unique | missing | ambiguous |
        |---|---|---|---|
    {chr(10).join(
        f"    | {row['model']} | {row.get('unique', 0):.1f}% | {row.get('missing', 0):.1f}% | {row.get('ambiguous', 0):.1f}% |"
        for row in pivot.iter_rows(named=True)
    )}

        **True distribution: 50% unique / 50% missing / 0% ambiguous**
        (recall: ambiguous is never a valid follower-side answer in this
        corpus -- Week 4 finding).

        - **LLaVA-NeXT** answers "missing" 97.6% of the time -- close to a
          constant classifier, largely ignoring case-specific content.
        - **Molmo** answers "ambiguous" 91.3% of the time -- the single
          most common answer choice is also the one that is *never*
          correct for this task, explaining its dramatic below-chance score.
        - **Qwen** shows the most balanced distribution of the three, but
          still leans toward "missing" (77.6%) versus the true 50/50 split.
        """
    )
    return (pivot,)


@app.cell
def _(follower_accuracy):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=follower_accuracy["model"].to_list(),
            y=follower_accuracy["accuracy"].to_list(),
            marker_color=["#4C72B0", "#DD8452", "#55A868"],
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Chance (50%)")
    fig.update_layout(
        title="follower_grounding accuracy by model (n=450, balanced)",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        template="plotly_white",
    )
    fig
    return (go,)


@app.cell
def _(go, pivot):
    fig2 = go.Figure()
    for status, color in [("unique", "#55A868"), ("missing", "#DD8452"), ("ambiguous", "#C44E52")]:
        fig2.add_trace(
            go.Bar(
                name=status,
                x=pivot["model"].to_list(),
                y=pivot[status].to_list() if status in pivot.columns else [0] * pivot.height,
            )
        )

    fig2.update_layout(
        barmode="stack",
        title="Response distribution by model (% of 450 follower_grounding cases)",
        yaxis_title="% of responses",
        template="plotly_white",
    )
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## Scope Limitation: giver_ambiguity

    The `giver_ambiguity` sub-experiment (440 cases) tests whether a
    model, shown the giver's map, recognizes that a landmark it's
    about to reference is ambiguous. All 440 cases share ground truth
    **"ambiguous"** -- the corpus contains no naturally-occurring
    negative examples for this specific task, so accuracy here
    (93-100% across models) does not measure real discrimination
    ability. It mostly reflects how often each model defaults to
    "Yes" regardless of the actual map. This is flagged as an honest
    scope limitation rather than a result to compare across models.

    ## Conclusions

    1. **None of the three tested open-weight VLMs reliably ground
       definite descriptions against map content** in the
       follower_grounding task -- all perform at or below the 50%
       chance level.
    2. **Each model exhibits a distinct, strong response bias** that
       dominates its measured accuracy more than genuine visual
       reasoning does (LLaVA-NeXT: near-constant "missing"; Molmo:
       near-constant "ambiguous", which is structurally never correct
       here; Qwen: the most varied, but still skewed).
    3. **Human listeners, by contrast, show varied, content-sensitive
       repair behavior** (Week 6) -- explicit rejection, repeated
       clarification-seeking, and active meaning negotiation -- a
       sharp qualitative contrast with the tested models' largely
       input-independent response patterns.
    4. **A corpus-design limitation constrains one sub-experiment**:
       giver-side ambiguity has no natural negative class in this
       corpus, limiting what conclusions can be drawn from that task
       alone.

    ## Future Work

    - Test additional models (particularly ones with explicit visual
      grounding/pointing training, e.g. larger Molmo variants) to see
      if response bias is a scale-dependent phenomenon.
    - Construct synthetic negative examples for giver_ambiguity (e.g.
      real giver maps known to be unambiguous) to enable a fair
      discrimination test.
    - Score the corpus's `moves/` dialogue-act annotations (identified
      but unexplored in Week 6) as a more reliable ground truth for
      human repair behavior than the keyword heuristic used here.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
