import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    return Path, mo


@app.cell
def _(mo):
    mo.md("""
    # 00 — Inspect raw HCRC Map Task corpus

    **Goal:** understand what's inside `data/raw/` before writing any ETL
    code — file naming, how a dialogue is represented, how transcripts link
    to maps, speaker IDs, and what annotation layers exist.

    **Rule:** this notebook only reads from `data/raw/`, never modifies it.
    """)
    return


@app.cell
def _(Path):
    RAW_DIR = Path("data/raw")
    RAW_DIR.exists()

    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    len(files)
    return RAW_DIR, files


@app.cell
def _(files):
    from collections import Counter

    ext_counts = Counter(p.suffix.lower() for p in files)
    ext_counts
    return


@app.cell
def _(RAW_DIR):
    files2 = list(RAW_DIR.rglob("*"));
    files2
    return


@app.cell
def _(mo):
    mo.md("""
    ## Findings

    - 128 transcript files under `Transcripts/`, named like `q1ec1.txt`
      (quad 1, eye-contact, conversation 1) or `q7nc7.txt` (no-eye-contact).
    - File format: 2 lines of export-tool metadata (`Found 2 args.` and
      `Observation: <id>; Result size: <n>; atts: 2`), then a `who\t$m`
      header, then tab-separated `role<TAB>utterance` rows.
    - Roles: `g` = giver (has the route), `f` = follower (has the map,
      no route).
    - `Result size` in line 2 matches the row count — useful as a
      validation check during ETL.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
