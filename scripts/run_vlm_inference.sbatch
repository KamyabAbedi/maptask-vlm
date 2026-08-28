#!/bin/bash
#SBATCH --job-name=maptask-vlm-qwen
#SBATCH --partition=p_csunivie_gres
#SBATCH --nodelist=vader
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vlm_job_%j.out
#SBATCH --error=logs/vlm_job_%j.err

set -euo pipefail

mkdir -p logs

source /etc/profile.d/modules.sh
module load python

cd "$SLURM_SUBMIT_DIR"
source .venv-gpu/bin/activate

export HF_HOME=/scratch/$USER/hf-cache
mkdir -p "$HF_HOME"

python scripts/run_vlm_inference.py --model qwen2.5-vl-7b