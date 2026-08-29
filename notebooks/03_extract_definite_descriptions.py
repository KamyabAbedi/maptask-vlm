import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    from pathlib import Path

    import polars as pl

    return Path, pl, re


@app.cell
def _(pl):
    utterances = pl.read_parquet("data/processed/utterances.parquet")
    utterances.shape
    return (utterances,)


@app.cell
def _(pl, utterances):
    sample = utterances.filter(pl.col("utterance").str.contains(r"\bthe \w+")).head(10)
    sample
    return


@app.cell
def _(pl):
    dialogues = pl.read_parquet("data/processed/dialogues.parquet")
    landmarks = pl.read_parquet("data/processed/landmarks.parquet")

    q1ec1_map = dialogues.filter(pl.col("dialogue_id") == "q1ec1")["map_id"][0]
    q1ec1_landmarks = landmarks.filter(pl.col("map_id") == q1ec1_map)["landmark_name"].to_list()

    q1ec1_map, q1ec1_landmarks
    return dialogues, landmarks, q1ec1_landmarks


@app.cell
def _(pl, q1ec1_landmarks, re, utterances):
    q1ec1_utterances = utterances.filter(pl.col("dialogue_id") == "q1ec1")

    matches = []
    for landmark_name in q1ec1_landmarks:
        hits = q1ec1_utterances.filter(
            pl.col("utterance").str.contains(re.escape(landmark_name))
        )
        for row in hits.iter_rows(named=True):
            matches.append(
                {
                    "dialogue_id": row["dialogue_id"],
                    "turn_id": row["turn_id"],
                    "speaker": row["speaker"],
                    "utterance": row["utterance"],
                    "landmark_name": landmark_name,
                }
            )

    matches_df = pl.DataFrame(matches).sort("turn_id")
    matches_df.shape, matches_df
    return (q1ec1_utterances,)


@app.cell
def _(pl, utterances):
    utterances.filter(pl.col("dialogue_id") == "q1ec1").filter(pl.col("turn_id") == 2)
    return


@app.cell
def _():
    # matches = []
    # for landmark_name in q1ec1_landmarks:
    #     pattern = r"\b" + re.escape(landmark_name) + r"\b"
    #     hits = q1ec1_utterances.filter(
    #         pl.col("utterance").str.contains(pattern)
    #     )
    #     for row in hits.iter_rows(named=True):
    #         matches.append(
    #             {
    #                 "dialogue_id": row["dialogue_id"],
    #                 "turn_id": row["turn_id"],
    #                 "speaker": row["speaker"],
    #                 "utterance": row["utterance"],
    #                 "landmark_name": landmark_name,
    #             }
    #         )

    # matches_df = pl.DataFrame(matches).sort("turn_id")
    # matches_df.shape, matches_df
    return


@app.cell
def _(pl, q1ec1_landmarks, q1ec1_utterances, re):
    matches_v2 = []
    for lm_name_v2 in q1ec1_landmarks:
        pattern_v2 = r"\b" + re.escape(lm_name_v2) + r"\b"
        hits_v2 = q1ec1_utterances.filter(
            pl.col("utterance").str.contains(pattern_v2)
        )
        for row_v2 in hits_v2.iter_rows(named=True):
            matches_v2.append(
                {
                    "dialogue_id": row_v2["dialogue_id"],
                    "turn_id": row_v2["turn_id"],
                    "speaker": row_v2["speaker"],
                    "utterance": row_v2["utterance"],
                    "landmark_name": lm_name_v2,
                }
            )

    matches_df_v2 = pl.DataFrame(matches_v2).sort("turn_id")
    matches_df_v2.shape, matches_df_v2.filter(pl.col("turn_id") == 2)
    return


@app.cell
def _():
    # mentions = find_landmark_mentions(utterances, dialogues, landmarks)
    return


@app.cell
def _():
    # from maptask_vlm.extraction import find_landmark_mentions

    # mentions = find_landmark_mentions(utterances, dialogues, landmarks)
    # mentions.shape
    return


@app.cell
def _():
    # from maptask_vlm.extraction import find_landmark_mentions

    # mentions_final = find_landmark_mentions(utterances, dialogues, landmarks)

    # mentions_output_path = Path("data/interim/definite_descriptions.parquet")
    # mentions_output_path.parent.mkdir(parents=True, exist_ok=True)
    # mentions_final.write_parquet(mentions_output_path)

    # mentions_output_path, mentions_final.shape
    return


@app.cell
def _(Path, dialogues, landmarks, utterances):
    from maptask_vlm.extraction import find_landmark_mentions

    mentions_final = find_landmark_mentions(utterances, dialogues, landmarks)

    mentions_output_path = Path("data/interim/definite_descriptions.parquet")
    mentions_output_path.parent.mkdir(parents=True, exist_ok=True)
    mentions_final.write_parquet(mentions_output_path)

    mentions_output_path, mentions_final.shape
    return (mentions_final,)


@app.cell
def _(dialogues, landmarks, mentions_final):
    mentions_with_status = mentions_final.join(
        dialogues.select("dialogue_id", "map_id"),
        on="dialogue_id",
        how="left",
    ).join(
        landmarks.select(
            "map_id", "landmark_name", "giver_referent_status", "follower_referent_status"
        ),
        on=["map_id", "landmark_name"],
        how="left",
    )

    mentions_with_status.shape, mentions_with_status.null_count()
    return (mentions_with_status,)


@app.cell
def _(mentions_with_status):
    mentions_with_status["follower_referent_status"].value_counts()
    return


@app.cell
def _(mentions_with_status):
    mentions_with_status["giver_referent_status"].value_counts()
    return


@app.cell
def _(Path, mentions_with_status):
    final_output_path = Path("data/interim/definite_descriptions.parquet")
    mentions_with_status.write_parquet(final_output_path)

    final_output_path, mentions_with_status.shape
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
