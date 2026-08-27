import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

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
      `Observation: <id>; Result size: <n>; atts: 2`), then a `who	$m`
      header, then tab-separated `role<TAB>utterance` rows.
    - Roles: `g` = giver (has the route), `f` = follower (has the map,
      no route).
    - `Result size` in line 2 matches the row count — useful as a
      validation check during ETL.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Corpus structure (Week 2 findings)

    The downloaded corpus has four parts under `data/raw/hcrc_maptask/`:

    - `Transcripts/` — 128 plain-text dialogue transcripts (`q1ec1.txt`
      style), tab-separated `role<TAB>utterance` rows.
    - `original_maps/maps/` — 32 GIF images: 16 base maps (`map0`–`map15`),
      each with a giver (`g`) and follower (`f`) version.
    - `maps/` — 8-per-page PDF scans of *completed* follower maps with
      hand-drawn routes overlaid; useful for accuracy analysis, **not**
      the clean input for VLM grounding experiments.
    - `annotations/maptaskv2-1/Data/` — the full NXT annotation set,
      including:
        - `corpus-resources/maptask-corpus.xml`: links every dialogue
          ID to its `map` ID and `quad`, e.g. `q1ec1` → `map="m12"`.
        - `corpus-resources/maptask-landmarks.xml`: per map, per
          landmark, exact appearance counts on giver's vs. follower's
          map (`giver_map_appears`, `follower_map_appears`).
        - `corpus-resources/maptask-participants.xml`: speaker
          demographics (age, gender, dialect region).
        - Many more annotation layers (`moves/`, `landmark-refs/`,
          `gaze/`, `disfluencies/`, etc.) not yet explored — may be
          relevant in later weeks.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Critical finding: referent status is already labeled

        `maptask-landmarks.xml` gives per-landmark appearance counts per map:

    ```xml
        <landmark name="parked van" follower_map_appears="1" giver_map_appears="2" .../>
    ```

        This means unique / missing / ambiguous referent status (the core
        label needed for Week 7's experiment cases) can be **derived directly
        from this file** — joined via `maptask-corpus.xml`'s dialogue→map
        mapping — rather than inferred by inspecting map images or running
        any vision model. Verified manually against `map0g.gif`/`map0f.gif`
        (Step 25/32): the file's counts for map `m0` exactly match what's
        visible in the images.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
