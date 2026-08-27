import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    return (Path,)


@app.cell
def _(Path):
    sample_path = Path("data/raw/hcrc_maptask/Transcripts/q1ec1.txt")
    raw_lines = sample_path.read_text().splitlines()
    len(raw_lines), raw_lines[:5]
    return (raw_lines,)


@app.cell
def _(raw_lines):
    sample_line = raw_lines[3]  # first data row
    sample_line, sample_line.split("\t")
    return


@app.cell
def _(raw_lines):
    header_line = raw_lines[2]
    header_line, header_line.split("\t")
    return


@app.cell
def _(raw_lines):
    def parse_transcript_lines(lines: list[str]) -> list[dict]:
        """Parse HCRC Map Task transcript lines into turn records.

        Skips the 2 metadata lines and the header, keeping only the
        role/utterance data rows.
        """
        data_lines = lines[3:]  # skip "Found 2 args.", "Observation: ...", header
        records = []
        for turn_id, line in enumerate(data_lines, start=1):
            parts = line.split("\t")
            speaker = parts[0].strip()
            utterance = parts[1].strip()
            records.append(
                {
                    "turn_id": turn_id,
                    "speaker": speaker,
                    "utterance": utterance,
                }
            )
        return records


    parsed = parse_transcript_lines(raw_lines)
    len(parsed), parsed[:3]
    return parse_transcript_lines, parsed


@app.cell
def _(parsed, raw_lines):
    import re

    def parse_observation_line(line: str) -> tuple[str, int]:
        """Extract (dialogue_id, expected_row_count) from the Observation line."""
        match = re.match(r"Observation:\s*(\S+);\s*Result size:\s*(\d+)", line)
        if not match:
            raise ValueError(f"Unexpected Observation line format: {line!r}")
        dialogue_id = match.group(1)
        expected_count = int(match.group(2))
        return dialogue_id, expected_count


    dialogue_id, expected_count = parse_observation_line(raw_lines[1])
    dialogue_id, expected_count, expected_count == len(parsed)
    return (parse_observation_line,)


@app.cell
def _(Path, parse_observation_line, parse_transcript_lines):
    TRANSCRIPTS_DIR = Path("data/raw/hcrc_maptask/Transcripts")

    def parse_transcript_file(path: Path) -> list[dict]:
        """Parse one transcript file into a list of turn records,
        each tagged with its dialogue_id, validated against the
        file's own declared Result size.
        """
        lines = path.read_text().splitlines()
        dialogue_id, expected_count = parse_observation_line(lines[1])

        turns = parse_transcript_lines(lines)

        if len(turns) != expected_count:
            raise ValueError(
                f"{path.name}: expected {expected_count} rows, got {len(turns)}"
            )

        for turn in turns:
            turn["dialogue_id"] = dialogue_id

        return turns


    all_files = sorted(TRANSCRIPTS_DIR.glob("*.txt"))
    len(all_files)
    return all_files, parse_transcript_file


@app.cell
def _(all_files, parse_transcript_file):
    all_turns = []
    failed_files = []

    for path in all_files:
        try:
            all_turns.extend(parse_transcript_file(path))
        except ValueError as e:
            failed_files.append((path.name, str(e)))

    len(all_turns), len(failed_files), failed_files
    return (all_turns,)


@app.cell
def _(all_turns):
    import polars as pl

    dialogue_turns = pl.DataFrame(all_turns).select(
        "dialogue_id", "turn_id", "speaker", "utterance"
    )
    dialogue_turns.shape, dialogue_turns.head(10)
    return dialogue_turns, pl


@app.cell
def _(dialogue_turns):
    # how many dialogues, and does it match 128?
    dialogue_turns["dialogue_id"].n_unique()
    return


@app.cell
def _(dialogue_turns):
    # what values does speaker take? (should only be g and f)
    dialogue_turns["speaker"].value_counts()
    return


@app.cell
def _(dialogue_turns, pl):
    # any empty utterances or nulls?
    dialogue_turns.filter(
        pl.col("utterance").is_null() | (pl.col("utterance") == "")
    ).height
    return


@app.cell
def _(dialogue_turns, pl):
    # urn_id always starts at 1 and increases by 1 with no gaps, per dialogue?
    gaps = (
        dialogue_turns
        .sort(["dialogue_id", "turn_id"])
        .group_by("dialogue_id")
        .agg(
            min_turn=pl.col("turn_id").min(),
            max_turn=pl.col("turn_id").max(),
            n_turns=pl.col("turn_id").len(),
        )
        .filter(
            (pl.col("min_turn") != 1) | (pl.col("max_turn") != pl.col("n_turns"))
        )
    )
    gaps.height, gaps
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(Path):
    from maptask_vlm.loaders import load_all_transcripts
    from maptask_vlm.transforms import turns_to_dataframe

    turns = load_all_transcripts()
    dialogue_turns_final = turns_to_dataframe(turns)

    output_path = Path("data/interim/dialogue_turns.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dialogue_turns_final.write_parquet(output_path)

    output_path, output_path.exists(), output_path.stat().st_size
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
