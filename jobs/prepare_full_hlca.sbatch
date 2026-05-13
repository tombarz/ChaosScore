#!/bin/bash
#SBATCH --job-name=prepare_hlca
#SBATCH --partition=power-general-shared-pool
#SBATCH --account=tzachiha-users_v2
#SBATCH --qos=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
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

module load miniconda3-24.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch200/yoren/conda_envs/chaosscore

test -f data/lung_data/hlca_core.h5ad
test -f external/scFoundation/model/OS_scRNA_gene_index.19264.tsv

python -m src.prepare_scfoundation_input \
  --input_h5ad data/lung_data/hlca_core.h5ad \
  --counts_source raw \
  --gene_symbol_field feature_name \
  --output_prefix data/processed/lung/full_hlca_core/scfoundation/hlca_core \
  --dataset_role mixed \
  --batch_size 20000

python -m src.create_train_val_test_splits \
  --input_h5ad data/lung_data/hlca_core.h5ad \
  --output_dir data/processed/lung/full_hlca_core/splits \
  --stratify_key ann_level_3 \
  --holdout_key donor_id \
  --seed 0

echo "Full HLCA preparation finished"
ls -lh data/processed/lung/full_hlca_core/scfoundation/
ls -lh data/processed/lung/full_hlca_core/splits/
