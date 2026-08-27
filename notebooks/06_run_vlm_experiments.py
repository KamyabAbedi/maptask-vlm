import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import polars as pl

    return Path, pl


@app.cell
def _(pl):
    experiment_cases = pl.read_parquet("data/processed/experiment_cases.parquet")
    experiment_cases.shape
    return (experiment_cases,)


@app.cell
def _(experiment_cases, pl):
    import random

    random.seed(42)

    def stratified_sample_by_dialogue(df: pl.DataFrame, n: int) -> pl.DataFrame:
        """Sample n rows, spreading picks across as many distinct dialogues
        as possible rather than allowing a few dialogues to dominate.
        """
        dialogue_ids = df["dialogue_id"].unique().to_list()
        random.shuffle(dialogue_ids)

        picked_rows = []
        remaining = n
        # Round-robin: take at most a few per dialogue per pass
        per_dialogue_cap = max(1, n // len(dialogue_ids) + 1)

        for dialogue_id in dialogue_ids:
            if remaining <= 0:
                break
            candidates = df.filter(pl.col("dialogue_id") == dialogue_id)
            take = min(per_dialogue_cap, candidates.height, remaining)
            picked_rows.append(candidates.sample(n=take, seed=42))
            remaining -= take

        return pl.concat(picked_rows).head(n)


    unique_pool = experiment_cases.filter(pl.col("referent_status") == "unique")
    missing_pool = experiment_cases.filter(pl.col("referent_status") == "missing")

    unique_sample = stratified_sample_by_dialogue(unique_pool, 225)
    missing_sample = stratified_sample_by_dialogue(missing_pool, 225)

    pilot_sample = pl.concat([unique_sample, missing_sample])

    pilot_sample.shape, pilot_sample["referent_status"].value_counts(), pilot_sample["dialogue_id"].n_unique()
    return (pilot_sample,)


@app.cell
def _(experiment_cases, pl):
    ambiguous_pool = experiment_cases.filter(pl.col("giver_referent_status") == "ambiguous")
    ambiguous_pool.shape, ambiguous_pool["dialogue_id"].n_unique()
    return (ambiguous_pool,)


@app.cell
def _(ambiguous_pool, pilot_sample, pl):
    pilot_ambiguous = ambiguous_pool  # use all 440, no further sampling needed

    pilot_full = pl.concat([
        pilot_sample.with_columns(pl.lit("follower_grounding").alias("experiment_type")),
        pilot_ambiguous.with_columns(pl.lit("giver_ambiguity").alias("experiment_type")),
    ])

    pilot_full.shape, pilot_full["experiment_type"].value_counts()
    return (pilot_full,)


@app.cell
def _(Path, pilot_full):
    pilot_output_path = Path("data/interim/pilot_cases.parquet")
    pilot_output_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_full.write_parquet(pilot_output_path)

    pilot_output_path, pilot_full.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
