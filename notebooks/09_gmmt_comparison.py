import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from maptask_vlm.gmmt import load_all_gmmt_reference_expressions

    return Path, load_all_gmmt_reference_expressions, mo, pl


@app.cell
def _(mo):
    mo.md("""
    # 09 — GMMT Comparison

    Compares this project's Week 6 keyword-heuristic repair detection
    against GMMT's (Li, Gatt & Poesio, LREC 2026) gold/LLM-annotated
    understanding states, and cross-checks the "old mill"/"mill wheel"
    lexical-variant finding.
    """)
    return


@app.cell
def _(load_all_gmmt_reference_expressions, pl):
    gmmt_records = load_all_gmmt_reference_expressions()
    gmmt_df = pl.DataFrame(gmmt_records)
    gmmt_df.shape
    return (gmmt_df,)


@app.cell
def _(gmmt_df):
    gmmt_df["understanding_state"].value_counts()
    return


@app.cell
def _(gmmt_df, pl):
    gmmt_df.filter(pl.col("concept_id") == "m12_old_mill")
    return


@app.cell
def _(gmmt_df, mo, pl):
    gmmt_misunderstood = gmmt_df.filter(pl.col("understanding_state") == "misunderstood")

    mo.md(
        f"""
        ## Comparison: Week 6 heuristic vs. GMMT gold labels

        - **Your Week 6 heuristic** flagged ~9% of missing-referent mentions
          as showing explicit repair, based on keyword matching.
        - **GMMT's actual measured rate**: only **{gmmt_misunderstood.height}
          of 13,077** REs ({gmmt_misunderstood.height/13077:.1%}) are true
          `misunderstood` cases (both sides believe they agree, but ground
          to different landmarks) -- most non-alignment is `pending`
          ({gmmt_df.filter(pl.col('understanding_state')=='pending').height},
          i.e. simply not yet grounded, not a failure of understanding).

        This confirms Healey et al. (2018)'s finding, cited in the GMMT
        paper: **sustained misunderstandings are rare** in collaborative
        dialogue -- participants repair continuously. Your Week 6 keyword
        heuristic was measuring something closer to "pending/repair
        activity," not true misunderstanding -- an important conceptual
        correction GMMT provides.
        """
    )
    return


@app.cell
def _():
    # mentions_with_context = pl.read_parquet("data/interim/mentions_with_context.parquet")

    # gmmt_missing_like = gmmt_df.filter(pl.col("understanding_state") != "aligned")
    # gmmt_misunderstood = gmmt_df.filter(pl.col("understanding_state") == "misunderstood")

    # mo.md(
    #     f"""
    #     ## Comparison: Week 6 heuristic vs. GMMT gold labels

    #     - **Your Week 6 heuristic** flagged ~9% of missing-referent mentions
    #       as showing explicit repair, based on keyword matching.
    #     - **GMMT's actual measured rate**: only **{gmmt_misunderstood.height}
    #       of 13,077** REs ({gmmt_misunderstood.height/13077:.1%}) are true
    #       `misunderstood` cases (both sides believe they agree, but ground
    #       to different landmarks) -- most non-alignment is `pending`
    #       ({gmmt_df.filter(pl.col('understanding_state')=='pending').height},
    #       i.e. simply not yet grounded, not a failure of understanding).

    #     This confirms Healey et al. (2018)'s finding, cited in the GMMT
    #     paper: **sustained misunderstandings are rare** in collaborative
    #     dialogue -- participants repair continuously. Your Week 6 keyword
    #     heuristic was measuring something closer to "pending/repair
    #     activity," not true misunderstanding -- an important conceptual
    #     correction GMMT provides.
    #     """
    # )
    return


@app.cell
def _(Path, gmmt_df):
    gmmt_output_path = Path("data/processed/gmmt_reference_expressions.parquet")
    gmmt_output_path.parent.mkdir(parents=True, exist_ok=True)
    gmmt_df.write_parquet(gmmt_output_path)

    gmmt_output_path, gmmt_df.shape
    return


@app.cell
def _():
    return


@app.cell
def _(pl):
    scored = pl.read_parquet("results/tables/evaluation_scored.parquet")
    scored.columns
    return (scored,)


@app.cell
def _(pl):
    pilot_cases_for_join = pl.read_parquet("data/interim/pilot_cases.parquet")
    pilot_cases_for_join.columns
    return (pilot_cases_for_join,)


@app.cell
def _(pl):
    gmmt_df_full = pl.read_parquet("data/processed/gmmt_reference_expressions.parquet")

    gmmt_df_full = gmmt_df_full.with_columns(
        pl.col("concept_id").str.replace(r"^m\d+_", "").str.replace_all("_", " ").alias("landmark_from_concept")
    )

    gmmt_summary = (
        gmmt_df_full
        .group_by(["dialogue_id", "landmark_from_concept"])
        .agg(
            pl.col("understanding_state").mode().first().alias("gmmt_dominant_state"),
            (pl.col("understanding_state") == "misunderstood").any().alias("gmmt_any_misunderstood"),
            pl.count().alias("gmmt_re_count"),
        )
    )

    gmmt_summary.shape
    return (gmmt_summary,)


@app.cell
def _(gmmt_summary, pilot_cases_for_join):
    pilot_with_gmmt = pilot_cases_for_join.join(
        gmmt_summary,
        left_on=["dialogue_id", "landmark"],
        right_on=["dialogue_id", "landmark_from_concept"],
        how="left",
    )

    pilot_with_gmmt["gmmt_dominant_state"].is_null().sum(), pilot_with_gmmt.height
    return (pilot_with_gmmt,)


@app.cell
def _(pilot_with_gmmt, pl, scored):
    scored_with_gmmt = scored.join(
        pilot_with_gmmt.select("case_id", "gmmt_dominant_state", "gmmt_any_misunderstood", "gmmt_re_count"),
        on="case_id",
        how="left",
    )

    scored_with_gmmt.filter(pl.col("experiment_type") == "follower_grounding").group_by(
        ["model", "gmmt_dominant_state"]
    ).agg(
        pl.col("correct").mean().alias("vlm_accuracy"),
        pl.count().alias("n"),
    ).sort(["model", "gmmt_dominant_state"])
    return (scored_with_gmmt,)


@app.cell
def _(pilot_with_gmmt, pl):
    pilot_with_gmmt.filter(pl.col("experiment_type") == "follower_grounding").group_by(
        ["referent_status", "gmmt_dominant_state"]
    ).agg(pl.count().alias("n")).sort(["referent_status", "gmmt_dominant_state"])
    return


@app.cell
def _(pl, scored_with_gmmt):
    scored_with_gmmt.filter(
        (pl.col("experiment_type") == "follower_grounding") &
        (pl.col("referent_status") == "missing")
    ).group_by(["model", "gmmt_dominant_state"]).agg(
        pl.col("correct").mean().alias("vlm_accuracy"),
        pl.count().alias("n"),
    ).sort(["model", "gmmt_dominant_state"])
    return


@app.cell
def _(mo):
    mo.md("""
    ## GMMT cross-check: does human grounding difficulty predict VLM accuracy?

    Joined GMMT's `understanding_state` onto 877/890 pilot cases
    (98.5% match rate). Within `referent_status == "missing"` cases only
    (holding the structural label constant, to isolate any effect from
    each model's `referent_status` guessing bias):

    | Model | pending (n=122) | aligned-despite-missing (n=65) |
    |---|---|---|
    | LLaVA-NeXT | 99.2% | 96.9% |
    | Molmo | 4.1% | 1.5% |
    | Qwen | 95.9% | **80.0%** |

    LLaVA and Molmo show no meaningful variation -- both are simply
    guessing "missing" mechanically, regardless of whether the
    landmark was truly ungrounded by humans (`pending`) or actually
    resolved via workaround (e.g. lexical variant; `aligned`).

    **Qwen shows a genuine, independent effect**: it performs notably
    worse (80.0% vs. 95.9%) specifically on cases where the landmark
    is structurally missing from the follower's map but humans still
    successfully grounded it (typically via lexical-variant
    substitution, e.g. "mill wheel" for "old mill"). This is exactly
    the subset requiring reasoning beyond simple map-lookup -- Qwen's
    "missing" bias, usually the correct default here, actively misleads
    it on precisely the cases where the human solution required
    creative reference resolution.
    """)
    return


@app.cell
def _(Path, scored_with_gmmt):
    gmmt_crosscheck_path = Path("results/tables/gmmt_crosscheck.parquet")
    gmmt_crosscheck_path.parent.mkdir(parents=True, exist_ok=True)
    scored_with_gmmt.write_parquet(gmmt_crosscheck_path)

    gmmt_crosscheck_path, scored_with_gmmt.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
