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


if __name__ == "__main__":
    app.run()
