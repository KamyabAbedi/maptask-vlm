import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from maptask_vlm.validation import validate_utterances

    return Path, mo, pl, validate_utterances


@app.cell
def _(mo):
    mo.md("""
    # 02 — Clean and validate: produce `data/processed/utterances.parquet`

    Load `data/interim/dialogue_turns.parquet`, validate it against
    `UtterancesSchema` (dialogue_id format, turn_id > 0, speaker in
    {g, f}, non-empty utterance), and write the validated table to
    `data/processed/utterances.parquet`.
    """)
    return


@app.cell
def _(Path, pl, validate_utterances):
    interim_path = Path("data/interim/dialogue_turns.parquet")
    utterances_raw = pl.read_parquet(interim_path)

    utterances_validated = validate_utterances(utterances_raw)

    output_path = Path("data/processed/utterances.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utterances_validated.write_parquet(output_path)

    output_path, utterances_validated.shape
    return (utterances_validated,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(Path):
    import xml.etree.ElementTree as ET

    corpus_xml_path = Path(
        "data/raw/hcrc_maptask/annotations/maptaskv2-1/Data/corpus-resources/maptask-corpus.xml"
    )
    tree = ET.parse(corpus_xml_path)
    root = tree.getroot()

    conv_elements = root.findall("conv")
    len(conv_elements)
    return ET, conv_elements


@app.cell
def _(conv_elements, pl):
    dialogue_records = []
    for conv in conv_elements:
        dialogue_records.append(
            {
                "dialogue_id": conv.get("id"),
                "quad": int(conv.get("quad")),
                "conversation": int(conv.get("conversation")),
                "eyecontact": conv.get("eyecontact") == "yes",
                "familiar": conv.get("familiar") == "yes",
                "map_id": conv.get("map"),
                "devscore": int(conv.get("devscore")),
            }
        )

    dialogues_df = pl.DataFrame(dialogue_records)
    dialogues_df.shape, dialogues_df.head(5)
    return (dialogues_df,)


@app.cell
def _(dialogues_df, utterances_validated):
    turns_ids = set(utterances_validated["dialogue_id"].unique())
    corpus_ids = set(dialogues_df["dialogue_id"].unique())

    only_in_turns = turns_ids - corpus_ids
    only_in_corpus = corpus_ids - turns_ids

    len(only_in_turns), len(only_in_corpus), only_in_turns, only_in_corpus
    return


@app.cell
def _(Path, dialogues_df):
    from maptask_vlm.validation import validate_dialogues

    dialogues_validated = validate_dialogues(dialogues_df)

    dialogues_output_path = Path("data/processed/dialogues.parquet")
    dialogues_output_path.parent.mkdir(parents=True, exist_ok=True)
    dialogues_validated.write_parquet(dialogues_output_path)

    dialogues_output_path, dialogues_validated.shape
    return


@app.cell
def _(ET, Path):
    landmarks_xml_path = Path(
        "data/raw/hcrc_maptask/annotations/maptaskv2-1/Data/corpus-resources/maptask-landmarks.xml"
    )
    landmarks_tree = ET.parse(landmarks_xml_path)
    landmarks_root = landmarks_tree.getroot()

    map_elements = landmarks_root.findall("map")
    len(map_elements)
    return (map_elements,)


@app.cell
def _(map_elements, pl):
    landmark_records = []
    for map_el in map_elements:
        map_id = map_el.get("id")
        for landmark_el in map_el.findall("landmark"):
            landmark_records.append(
                {
                    "map_id": map_id,
                    "landmark_name": landmark_el.get("name"),
                    "landmark_id": landmark_el.get("id"),
                    "giver_map_appears": int(landmark_el.get("giver_map_appears")),
                    "follower_map_appears": int(landmark_el.get("follower_map_appears")),
                }
            )

    landmarks_df = pl.DataFrame(landmark_records)
    landmarks_df.shape, landmarks_df.head(5)
    return (landmarks_df,)


@app.cell
def _(landmarks_df, pl):
    def referent_status(giver_count: int, follower_count: int) -> str:
        """Classify a landmark's referent status from the follower's
        perspective (the listener trying to ground a definite description).
        """
        if follower_count == 0:
            return "missing"
        if follower_count > 1:
            return "ambiguous"
        return "unique"


    landmarks_df2 = landmarks_df.with_columns(
        pl.struct(["giver_map_appears", "follower_map_appears"])
        .map_elements(
            lambda row: referent_status(row["giver_map_appears"], row["follower_map_appears"]),
            return_dtype=pl.String,
        )
        .alias("follower_referent_status")
    )

    landmarks_df2["follower_referent_status"].value_counts()
    return landmarks_df2, referent_status


@app.cell
def _(landmarks_df, pl, referent_status):
    landmarks_df3 = landmarks_df.with_columns(
        pl.struct(["giver_map_appears", "follower_map_appears"])
        .map_elements(
            lambda row: referent_status(row["follower_map_appears"], row["giver_map_appears"]),
            return_dtype=pl.String,
        )
        .alias("giver_referent_status")
    )

    landmarks_df3["giver_referent_status"].value_counts()
    return (landmarks_df3,)


@app.cell
def _(landmarks_df):
    landmarks_df
    return


@app.cell
def _(landmarks_df2):
    landmarks_df2
    return


@app.cell
def _(landmarks_df3):
    landmarks_df3
    return


@app.cell
def _(landmarks_df, pl, referent_status):
    landmarks_final = landmarks_df.with_columns(
        pl.struct(["giver_map_appears", "follower_map_appears"])
        .map_elements(
            lambda row: referent_status(row["giver_map_appears"], row["follower_map_appears"]),
            return_dtype=pl.String,
        )
        .alias("follower_referent_status"),
        pl.struct(["giver_map_appears", "follower_map_appears"])
        .map_elements(
            lambda row: referent_status(row["follower_map_appears"], row["giver_map_appears"]),
            return_dtype=pl.String,
        )
        .alias("giver_referent_status"),
    )

    landmarks_final.select(
        "map_id", "landmark_name", "giver_map_appears", "follower_map_appears",
        "giver_referent_status", "follower_referent_status"
    ).head(10)
    return (landmarks_final,)


@app.cell
def _(Path, landmarks_final):
    from maptask_vlm.validation import validate_landmarks

    landmarks_validated = validate_landmarks(
        landmarks_final.select(
            "map_id", "landmark_name", "landmark_id",
            "giver_map_appears", "follower_map_appears",
            "giver_referent_status", "follower_referent_status",
        )
    )

    landmarks_output_path = Path("data/processed/landmarks.parquet")
    landmarks_output_path.parent.mkdir(parents=True, exist_ok=True)
    landmarks_validated.write_parquet(landmarks_output_path)

    landmarks_output_path, landmarks_validated.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
