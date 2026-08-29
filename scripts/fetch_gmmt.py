"""Fetch the GMMT (Grounded Misunderstandings in MapTask) dataset.

GMMT (Li, Gatt & Poesio, LREC 2026) provides perspectivist annotations
for reference expressions in the HCRC Map Task corpus: for each RE, the
giver's intended landmark and the follower's interpreted landmark, plus
an understanding_state (aligned / pending / misunderstood).

Source: https://github.com/chnln/grounded-misunderstandings-in-maptask
License: CC-BY-4.0
"""

import zipfile
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent

url = "https://github.com/chnln/grounded-misunderstandings-in-maptask/archive/refs/heads/main.zip"

raw_dir = PROJECT_ROOT / "data" / "raw"
zip_path = raw_dir / "gmmt.zip"
extract_dir = raw_dir / "gmmt"

raw_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

print("Downloading GMMT dataset...")

with httpx.stream("GET", url, follow_redirects=True) as response:
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.writelines(response.iter_bytes())

print("Extracting GMMT dataset...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

zip_path.unlink()

print(f"Done. GMMT extracted to: {extract_dir}")
