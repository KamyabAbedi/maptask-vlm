import tarfile
from pathlib import Path

import httpx

# Project root: one level above /scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent

url = "https://groups.inf.ed.ac.uk/maptask/interface/maps.tar.gz"

raw_dir = PROJECT_ROOT / "data" / "raw"
tar_path = raw_dir / "original-maps.tar.gz"
extract_dir = raw_dir / "hcrc_maptask" / "original_maps"

# Create directories
raw_dir.mkdir(parents=True, exist_ok=True)
extract_dir.mkdir(parents=True, exist_ok=True)

# Download
print("Downloading original maps...")

with httpx.stream("GET", url, follow_redirects=True) as response:
    response.raise_for_status()

    with open(tar_path, "wb") as f:
        f.writelines(response.iter_bytes())

# Extract
print("Extracting original maps...")

with tarfile.open(tar_path, "r:") as tar:
    tar.extractall(extract_dir)

# Remove archive
tar_path.unlink()

print(f"Done. Original maps extracted to: {extract_dir}")
