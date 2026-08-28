#!/bin/bash
#SBATCH --job-name=maptask-vlm-qwen
#SBATCH --partition=p_csunivie_gres
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vlm_job_%j.out
#SBATCH --error=logs/vlm_job_%j.err

set -euo pipefail

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

source /etc/profile.d/modules.sh

export ENV_MODE="permanent"
export ENV_NAME="maptask-vlm-gpu"
module load python/3.11.11

pip install --quiet -r requirements-cluster.txt
pip install --quiet -e .

export HF_HOME="$HOME/hf-cache"
mkdir -p "$HF_HOME"

python scripts/run_vlm_inference.py --model "${MODEL_NAME:-qwen2.5-vl-7b}"

module purge
