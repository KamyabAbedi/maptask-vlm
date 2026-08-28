# Running VLM experiments on the University of Vienna Slurm cluster

This project's core pipeline (Weeks 1-7) runs locally via `uv`. The VLM
experiments (Week 8) need a GPU and run separately, using
`requirements-cluster.txt` instead of the `uv`-managed environment.

## One-time setup on the cluster

1. Clone this repo onto the cluster (login node):
```bash
   git clone https://github.com/KamyabAbedi/maptask-vlm.git
   cd maptask-vlm
```

2. Create a Python virtual environment for the GPU dependencies
   (kept separate from the `uv`-managed `.venv` used locally):
```bash
   module load python/3.12
   python -m venv .venv-gpu
   source .venv-gpu/bin/activate
   pip install -r requirements-cluster.txt
```

3. Re-fetch the corpus (raw data is gitignored, not in the repo):
```bash
   pip install httpx  # needed by the fetch scripts
   python scripts/fetch_data.py
   python scripts/fetch_giver_follower_maps.py
   python scripts/fetch_annotations.py
```

4. Rebuild the processed datasets, OR copy them from your local machine
   (faster -- these are small parquet files):
```bash
   scp data/processed/*.parquet data/interim/pilot_cases.parquet \
       yourcs@slurm.cs.univie.ac.at:~/maptask-vlm/data/processed/
```

## Running the inference job

See `scripts/run_vlm_inference.sbatch` (Slurm batch script) and
`scripts/run_vlm_inference.py` (the actual inference loop).

```bash
sbatch scripts/run_vlm_inference.sbatch
squeue -u yourcs          # check job status
tail -f logs/vlm_job_*.out  # watch progress
```

## Model weights caching

By default, Hugging Face downloads model weights to `~/.cache/huggingface`,
which may be small/quota-limited on the login node's home directory.
Set `HF_HOME` to a larger scratch location before running, e.g.:

```bash
export HF_HOME=/scratch/yourcs/hf-cache
```