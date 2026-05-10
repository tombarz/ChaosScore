#!/bin/bash
#SBATCH --job-name=train_hlca_mgp
#SBATCH --partition=gpu-general-pool
#SBATCH --account=tzachiha-users_v2
#SBATCH --qos=public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch200/yoren/slurm_logs/%x_%j.out
#SBATCH --error=/scratch200/yoren/slurm_logs/%x_%j.err

set -euo pipefail

cd /scratch200/yoren/ChaosScore
mkdir -p /scratch200/yoren/slurm_logs

export HOME=/scratch200/yoren/home
export CONDA_ENVS_PATH=/scratch200/yoren/conda_envs
export CONDA_PKGS_DIRS=/scratch200/yoren/conda_pkgs
export PIP_CACHE_DIR=/scratch200/yoren/pip_cache
export PYTHONPATH=/scratch200/yoren/ChaosScore:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

module load miniconda3-24.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch200/yoren/conda_envs/chaosscore

RUN_NAME="full_hlca_mgp_mask10_attention_bs8_s0_${SLURM_JOB_ID}"

echo "$(date) Python: $(which python)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

test -f data/processed/lung/full_hlca_core/scfoundation/hlca_core.counts_19264.npz
test -f data/processed/lung/full_hlca_core/splits/split_assignments.csv.gz

echo "$(date) Starting full HLCA masked-gene training: ${RUN_NAME}"
python -u scripts/train_scfoundation_conditioned_masked_task.py \
  --prepared_prefix data/processed/lung/full_hlca_core/scfoundation/hlca_core \
  --cell_type_key ann_level_3 \
  --split_assignments data/processed/lung/full_hlca_core/splits/split_assignments.csv.gz \
  --train_split train \
  --eval_splits validation \
  --save_dir "outputs/experiments/${RUN_NAME}" \
  --device cuda \
  --epochs 5 \
  --batch_size 8 \
  --num_workers 4 \
  --pooling attention \
  --loss_type mse \
  --mask_ratio 0.10 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --freeze_encoder \
  --use_depth_covariate \
  --log_every_batches 100 \
  --checkpoint_every_batches 1000 \
  --save_epoch_checkpoints

echo "$(date) Training finished: ${RUN_NAME}"
