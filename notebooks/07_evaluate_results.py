import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import polars as pl

    return Path, json, mo, pl


@app.cell
def _(Path, json, pl):
    def load_predictions(path: Path) -> pl.DataFrame:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        return pl.DataFrame(records)

    pred_dir = Path("results/predictions")
    qwen_preds = load_predictions(pred_dir / "vlm_predictions_qwen2.5-vl-7b.jsonl")
    molmo_preds = load_predictions(pred_dir / "vlm_predictions_molmo-7b-d.jsonl")
    llava_preds = load_predictions(pred_dir / "vlm_predictions_llava-next-7b.jsonl")

    qwen_preds.shape, molmo_preds.shape, llava_preds.shape
    return llava_preds, molmo_preds, qwen_preds


@app.cell
def _(llava_preds, molmo_preds, qwen_preds):
    for name, df in [("qwen", qwen_preds), ("molmo", molmo_preds), ("llava", llava_preds)]:
        print(f"--- {name} ---")
        for resp in df["raw_response"].head(5).to_list():
            print(repr(resp))
        print()
    return


@app.cell
def _(pl):
    from maptask_vlm.evaluation import parse_response_letter

    def add_parsed_status(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("raw_response")
            .map_elements(parse_response_letter, return_dtype=pl.String)
            .alias("predicted_status")
        )

    # qwen_parsed = add_parsed_status(qwen_preds)
    # molmo_parsed = add_parsed_status(molmo_preds)
    # llava_parsed = add_parsed_status(llava_preds)

    # llava_parsed["predicted_status"].value_counts()
    return


@app.cell
def _(llava_parsed, pl):
    llava_parsed.filter(pl.col("predicted_status").is_null())["raw_response"].head(10).to_list()
    return


@app.cell
def _(llava_preds, molmo_preds, pl, qwen_preds):
    from maptask_vlm.evaluation import parse_predicted_status

    def add_parsed_status_v2(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.struct(["raw_response", "experiment_type"])
            .map_elements(
                lambda row: parse_predicted_status(row["raw_response"], row["experiment_type"]),
                return_dtype=pl.String,
            )
            .alias("predicted_status")
        )

    qwen_final = add_parsed_status_v2(qwen_preds)
    molmo_final = add_parsed_status_v2(molmo_preds)
    llava_final = add_parsed_status_v2(llava_preds)

    llava_final.group_by("experiment_type").agg(
        pl.col("predicted_status").is_null().sum().alias("null_count"),
        pl.count().alias("total"),
    )
    return add_parsed_status_v2, llava_final, molmo_final, qwen_final


@app.cell
def _(llava_final, pl):
    llava_final.filter(pl.col("experiment_type") == "follower_grounding")["predicted_status"].value_counts()
    return


@app.cell
def _(llava_final, molmo_final, pl, qwen_final):
    pilot_cases = pl.read_parquet("data/interim/pilot_cases.parquet")

    def compute_accuracy(model_preds: pl.DataFrame, model_name: str) -> pl.DataFrame:
        joined = model_preds.join(
            pilot_cases.select("case_id", "experiment_type", "referent_status", "giver_referent_status"),
            on="case_id",
            how="left",
            suffix="_truth",
        )

        joined = joined.with_columns(
            pl.when(pl.col("experiment_type") == "follower_grounding")
            .then(pl.col("referent_status"))
            .otherwise(pl.col("giver_referent_status"))
            .alias("ground_truth")
        )

        joined = joined.with_columns(
            (pl.col("predicted_status") == pl.col("ground_truth")).alias("correct")
        )

        return joined.with_columns(pl.lit(model_name).alias("model"))

    qwen_scored = compute_accuracy(qwen_final, "qwen2.5-vl-7b")
    molmo_scored = compute_accuracy(molmo_final, "molmo-7b-d")
    llava_scored = compute_accuracy(llava_final, "llava-next-7b")

    all_scored = pl.concat([qwen_scored, molmo_scored, llava_scored])

    all_scored.group_by("model").agg(
        pl.col("correct").mean().alias("overall_accuracy")
    ).sort("model")
    return all_scored, compute_accuracy, pilot_cases, qwen_scored


@app.cell
def _(all_scored, pl):
    all_scored.group_by(["model", "experiment_type"]).agg(
        pl.col("correct").mean().alias("accuracy"),
        pl.count().alias("n"),
    ).sort(["experiment_type", "model"])
    return


@app.cell
def _(llava_final, pl):
    llava_final.filter(pl.col("experiment_type") == "giver_ambiguity")["predicted_status"].value_counts()
    return


@app.cell
def _(pilot_cases, pl, qwen_scored):
    qwen_scored.filter(pl.col("experiment_type") == "follower_grounding").height
    pilot_cases.filter(pl.col("experiment_type") == "follower_grounding").height
    qwen_scored["case_id"].n_unique()
    qwen_scored.height
    return


@app.cell
def _(pilot_cases, pl, qwen_scored):
    a = qwen_scored.filter(pl.col("experiment_type") == "follower_grounding").height
    b = pilot_cases.filter(pl.col("experiment_type") == "follower_grounding").height
    c = qwen_scored["case_id"].n_unique()
    d = qwen_scored.height
    a, b, c, d
    return


@app.cell
def _(pilot_cases):
    pilot_cases["case_id"].n_unique(), pilot_cases.height
    return


@app.cell
def _(pilot_cases, pl):
    duplicates = (
        pilot_cases.group_by("case_id")
        .agg(pl.count().alias("n"))
        .filter(pl.col("n") > 1)
        .sort("n", descending=True)
    )
    duplicates
    return


@app.cell
def _(pilot_cases, pl):
    pilot_cases.filter(pl.col("case_id") == "q1ec1_34_fenced_meadow")
    return


@app.cell
def _(llava_preds, molmo_preds, pl, qwen_preds):
    def fix_case_id(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            (pl.col("experiment_type") + "__" + pl.col("case_id")).alias("case_id")
        )

    qwen_preds_fixed = fix_case_id(qwen_preds)
    molmo_preds_fixed = fix_case_id(molmo_preds)
    llava_preds_fixed = fix_case_id(llava_preds)

    qwen_preds_fixed["case_id"].n_unique(), qwen_preds_fixed.height
    return llava_preds_fixed, molmo_preds_fixed, qwen_preds_fixed


@app.cell
def _(
    add_parsed_status_v2,
    compute_accuracy,
    llava_preds_fixed,
    molmo_preds_fixed,
    pl,
    qwen_preds_fixed,
):
    qwen_final_v2 = add_parsed_status_v2(qwen_preds_fixed)
    molmo_final_v2 = add_parsed_status_v2(molmo_preds_fixed)
    llava_final_v2 = add_parsed_status_v2(llava_preds_fixed)

    qwen_scored_v2 = compute_accuracy(qwen_final_v2, "qwen2.5-vl-7b")
    molmo_scored_v2 = compute_accuracy(molmo_final_v2, "molmo-7b-d")
    llava_scored_v2 = compute_accuracy(llava_final_v2, "llava-next-7b")

    all_scored_v2 = pl.concat([qwen_scored_v2, molmo_scored_v2, llava_scored_v2])

    all_scored_v2["case_id"].n_unique(), all_scored_v2.height
    return (
        all_scored_v2,
        llava_final_v2,
        molmo_final_v2,
        qwen_final_v2,
        qwen_scored_v2,
    )


@app.cell
def _(all_scored_v2, pl):
    all_scored_v2.group_by(["model", "experiment_type"]).agg(
        pl.col("correct").mean().alias("accuracy"),
        pl.count().alias("n"),
    ).sort(["experiment_type", "model"])
    return


@app.cell
def _(qwen_scored_v2):
    qwen_scored_v2.select("case_id", "predicted_status", "ground_truth", "correct").head(10)
    return


@app.cell
def _(pl):
    pilot_cases_v2 = pl.read_parquet("data/interim/pilot_cases.parquet")
    pilot_cases_v2["case_id"].n_unique(), pilot_cases_v2.height
    return (pilot_cases_v2,)


@app.cell
def _(llava_final_v2, molmo_final_v2, pilot_cases_v2, pl, qwen_final_v2):
    def compute_accuracy_v2(model_preds: pl.DataFrame, model_name: str) -> pl.DataFrame:
        joined = model_preds.join(
            pilot_cases_v2.select("case_id", "experiment_type", "referent_status", "giver_referent_status"),
            on="case_id",
            how="left",
            suffix="_truth",
        )

        joined = joined.with_columns(
            pl.when(pl.col("experiment_type") == "follower_grounding")
            .then(pl.col("referent_status"))
            .otherwise(pl.col("giver_referent_status"))
            .alias("ground_truth")
        )

        joined = joined.with_columns(
            (pl.col("predicted_status") == pl.col("ground_truth")).alias("correct")
        )

        return joined.with_columns(pl.lit(model_name).alias("model"))


    qwen_scored_v3 = compute_accuracy_v2(qwen_final_v2, "qwen2.5-vl-7b")
    molmo_scored_v3 = compute_accuracy_v2(molmo_final_v2, "molmo-7b-d")
    llava_scored_v3 = compute_accuracy_v2(llava_final_v2, "llava-next-7b")

    all_scored_v3 = pl.concat([qwen_scored_v3, molmo_scored_v3, llava_scored_v3])

    all_scored_v3.group_by(["model", "experiment_type"]).agg(
        pl.col("correct").mean().alias("accuracy"),
        pl.count().alias("n"),
    ).sort(["experiment_type", "model"])
    return all_scored_v3, molmo_scored_v3, qwen_scored_v3


@app.cell
def _(pl, qwen_scored_v3):
    qwen_scored_v3.filter(pl.col("experiment_type") == "giver_ambiguity")["ground_truth"].value_counts()
    return


@app.cell
def _(molmo_scored_v3, pl):
    molmo_scored_v3.filter(pl.col("experiment_type") == "follower_grounding")["predicted_status"].value_counts()
    return


@app.cell
def _(all_scored_v3, pl):
    behavior_summary = (
        all_scored_v3
        .filter(pl.col("experiment_type") == "follower_grounding")
        .group_by(["model", "predicted_status"])
        .agg(pl.count().alias("n"))
        .sort(["model", "predicted_status"])
    )
    behavior_summary
    return


@app.cell
def _(Path, all_scored_v3, pl):
    eval_output_path = Path("results/tables/evaluation_scored.parquet")
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_scored_v3.write_parquet(eval_output_path)

    summary_output_path = Path("results/tables/accuracy_summary.csv")
    all_scored_v3.group_by(["model", "experiment_type"]).agg(
        pl.col("correct").mean().alias("accuracy"),
        pl.count().alias("n"),
    ).sort(["experiment_type", "model"]).write_csv(summary_output_path)

    eval_output_path, summary_output_path
    return


@app.cell
def _(mo):
    mo.md("""
    ## Week 9 findings: VLM referent-grounding evaluation

    ### Data quality fix
    Found and fixed a case_id collision bug: 17 landmark mentions
    qualified for *both* the follower_grounding and giver_ambiguity
    samples (same mention, unique from the follower's side, ambiguous
    from the giver's side), causing case_id to not be a valid unique
    key. Fixed by prefixing case_id with experiment_type.

    ### follower_grounding results (binary, balanced 225 unique / 225 missing)

    | Model | Accuracy | Dominant answer |
    |---|---|---|
    | Qwen2.5-VL-7B | 54.7% | missing (78%) |
    | LLaVA-NeXT-7B | 49.9% | missing (98%) |
    | Molmo-7B-D | 5.6% | ambiguous (91%) |

    All three models show near-chance or below-chance accuracy, driven
    by strong, largely input-independent response biases rather than
    genuine per-case visual grounding:
    - LLaVA-NeXT answers "missing" for 98% of cases regardless of
      content -- close to a constant classifier.
    - Molmo answers "ambiguous" for 91% of cases -- but "ambiguous" is
      *never* a valid follower-side answer in this corpus (Week 4
      finding: follower maps never have duplicate landmarks), so this
      single bias alone drives its dramatic below-chance score.
    - Qwen shows the most variation across the three answer choices,
      but still leans toward "missing" (78%) versus the true 50/50 split.

    ### giver_ambiguity results (limitation, not a real discrimination test)

    All 440 cases have ground truth "ambiguous" -- there are no
    naturally-occurring negative examples in this corpus (Week 8
    design note). Accuracy here (93-100%) mostly reflects how often
    each model defaults to "Yes," not real discrimination ability.
    Molmo's 93% (occasionally saying "No") is arguably more informative
    than Qwen/LLaVA's 100%, since it's the only model showing any
    answer variation at all on this sub-task.

    ### Conclusion
    None of the three tested open-weight VLMs reliably ground definite
    descriptions against actual map content in this task; each exhibits
    a distinct, strong response bias that dominates its measured
    accuracy more than genuine visual reasoning does.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
