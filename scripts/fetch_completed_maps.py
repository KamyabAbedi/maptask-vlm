import zipfile
from pathlib import Path

import httpx

# Project root: one level above /scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent

url = "https://groups.inf.ed.ac.uk/maptask/maps/allmaps.zip"

raw_dir = PROJECT_ROOT / "data" / "raw"
zip_path = raw_dir / "maps.zip"
extract_dir = raw_dir / "hcrc_maptask" / "Maps"

# Create directories
raw_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

# Download
print("Downloading completed maps...")

with httpx.stream("GET", url, follow_redirects=True) as response:
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        f.writelines(response.iter_bytes())

# Extract
print("Extracting completed maps...")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

# Remove ZIP
zip_path.unlink()

print(f"Done. Completed maps extracted to: {extract_dir}")
