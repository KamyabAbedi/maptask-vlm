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
    mentions_with_context = pl.read_parquet("data/interim/mentions_with_context.parquet")
    mentions_with_context.shape
    return (mentions_with_context,)


@app.cell
def _(mentions_with_context, pl):
    import re

    def map_id_to_number(map_id: str) -> str:
        """'m12' -> '12'"""
        match = re.match(r"^m(\d+)$", map_id)
        if not match:
            raise ValueError(f"Unexpected map_id format: {map_id!r}")
        return match.group(1)


    experiment_cases = (
        mentions_with_context
        .with_columns(
            pl.col("map_id")
            .map_elements(map_id_to_number, return_dtype=pl.String)
            .alias("map_number")
        )
        .with_columns(
            ("data/raw/hcrc_maptask/original_maps/maps/map" + pl.col("map_number") + "f.gif")
            .alias("map_path"),
            pl.col("follower_referent_status").alias("referent_status"),
            pl.col("landmark_name").alias("landmark"),
            pl.col("utterance").alias("definite_description_context"),
            pl.col("next_utterance").alias("human_response"),
        )
        .with_columns(
            (pl.col("dialogue_id") + "_" + pl.col("turn_id").cast(pl.String) + "_" + pl.col("landmark").str.replace_all(" ", "_"))
            .alias("case_id")
        )
        .select(
            "case_id",
            "dialogue_id",
            "turn_id",
            "map_path",
            "landmark",
            "definite_description_context",
            "referent_status",
            "giver_referent_status",
            "human_response",
        )
    )

    experiment_cases.shape, experiment_cases.head(5)
    return (experiment_cases,)


@app.cell
def _(Path, experiment_cases, pl):
    experiment_cases_with_check = experiment_cases.with_columns(
        pl.col("map_path").map_elements(
            lambda p: Path(p).exists(), return_dtype=pl.Boolean
        ).alias("map_file_exists")
    )

    experiment_cases_with_check["map_file_exists"].value_counts()
    return


@app.cell
def _(Path, experiment_cases_with_giver_map):
    from maptask_vlm.validation import validate_experiment_cases

    # experiment_cases_validated = validate_experiment_cases(experiment_cases)
    experiment_cases_validated = validate_experiment_cases(experiment_cases_with_giver_map)

    cases_output_path = Path("data/processed/experiment_cases.parquet")
    cases_output_path.parent.mkdir(parents=True, exist_ok=True)
    experiment_cases_validated.write_parquet(cases_output_path)

    cases_output_path, experiment_cases_validated.shape
    return


@app.cell
def _(experiment_cases, pl):
    experiment_cases_with_giver_map = experiment_cases.with_columns(
        pl.col("map_path").str.replace("f.gif", "g.gif").alias("giver_map_path")
    )

    experiment_cases_with_giver_map.select("map_path", "giver_map_path").head(3)
    return (experiment_cases_with_giver_map,)


@app.cell
def _(Path, experiment_cases_with_giver_map, pl):
    experiment_cases_with_giver_map.with_columns(
        pl.col("giver_map_path").map_elements(
            lambda p: Path(p).exists(), return_dtype=pl.Boolean
        ).alias("giver_map_file_exists")
    )["giver_map_file_exists"].value_counts()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
