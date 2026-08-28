"""Run VLM inference over the pilot experiment cases.

Usage (on the cluster, inside the GPU venv):
    python scripts/run_vlm_inference.py --model qwen2.5-vl-7b

Saves results incrementally to results/predictions/, one row per
(case_id, model) pair, so a crash or job timeout doesn't lose
completed work -- on restart, already-completed cases are skipped.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from maptask_vlm.models import InternVLModel, MolmoModel, QwenVLModel
from maptask_vlm.prompts import (
    build_follower_grounding_prompt,
    build_giver_ambiguity_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_CASES_PATH = PROJECT_ROOT / "data" / "interim" / "pilot_cases.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "predictions"

MODEL_REGISTRY = {
    "qwen2.5-vl-7b": QwenVLModel,
    "molmo-7b-d": MolmoModel,
    "internvl2.5-8b": InternVLModel,
}


def build_prompt_and_image(row: dict) -> tuple[str, str]:
    """Return (prompt, image_path) for one experiment case row."""
    if row["experiment_type"] == "follower_grounding":
        prompt = build_follower_grounding_prompt(row["definite_description_context"])
        image_path = row["map_path"]
    else:  # giver_ambiguity
        prompt = build_giver_ambiguity_prompt(row["landmark"])
        image_path = row["giver_map_path"]
    return prompt, image_path


def load_completed_case_ids(output_path: Path) -> set[str]:
    """Read already-saved SUCCESSFUL results (if any) to support resuming.
    Cases that previously errored (raw_response is null) are excluded, so
    they get retried rather than permanently skipped.
    """
    if not output_path.exists():
        return set()
    existing = pl.read_ndjson(output_path)
    successful = existing.filter(pl.col("raw_response").is_not_null())
    return set(successful["case_id"].to_list())


def main(model_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"vlm_predictions_{model_name}.jsonl"

    cases = pl.read_parquet(PILOT_CASES_PATH)
    completed_ids = load_completed_case_ids(output_path)
    print(f"Loaded {cases.height} cases, {len(completed_ids)} already completed.")

    print(f"Loading model: {model_name}...")
    model = MODEL_REGISTRY[model_name]()
    print("Model loaded.")

    with open(output_path, "a") as f:
        for i, row in enumerate(cases.iter_rows(named=True)):
            if row["case_id"] in completed_ids:
                continue

            prompt, image_path = build_prompt_and_image(row)

            try:
                response = model.answer(image_path, prompt)
            except Exception as e:  # noqa: BLE001 -- log and continue, don't lose the whole job
                response = None
                print(f"ERROR on case {row['case_id']}: {e}")

            result = {
                "case_id": row["case_id"],
                "model": model_name,
                "experiment_type": row["experiment_type"],
                "prompt": prompt,
                "raw_response": response,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{cases.height} cases done")

    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    args = parser.parse_args()
    main(args.model)
