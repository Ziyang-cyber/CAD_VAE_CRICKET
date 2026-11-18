#!/bin/bash
#SBATCH --job-name=cadvae_cricket_lri
#SBATCH --output=/mnt/shared/scratch/zzhang/hawaii_300/CAD_VAE_cricket/cricket_block_hyper/cadvae_cricket_lri_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/zzhang/hawaii_300/CAD_VAE_cricket/cricket_block_hyper/cadvae_cricket_lri_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-15
#SBATCH --time=64:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

set -euo pipefail

SCAFFOLDS=(10 1)
BLOCK_IDS=(176 136)
BETA_VALUES=(1)
LRI_VALUES=(0 20 40 60 80 100 500 1000)

MAX_EPOCHS=200
BATCH_SIZE=32
SEED=1024
TARGET_COLUMN=venation
BETA_WARMUP_EPOCHS=50
LAMBDA_CLS=500
LAMBDA_TC=0
LAMBDA_CMI=500

# Each task runs five folds.
N_BETA=${#BETA_VALUES[@]}
N_LRI=${#LRI_VALUES[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
CONFIGS_PER_BLOCK=$((N_BETA * N_LRI))

BLOCK_INDEX=$((TASK_ID / CONFIGS_PER_BLOCK))
CONFIG_INDEX=$((TASK_ID % CONFIGS_PER_BLOCK))
BETA_INDEX=$((CONFIG_INDEX / N_LRI))
LRI_INDEX=$((CONFIG_INDEX % N_LRI))

SCAFFOLD=${SCAFFOLDS[$BLOCK_INDEX]}
BLOCK_ID=${BLOCK_IDS[$BLOCK_INDEX]}
BETA=${BETA_VALUES[$BETA_INDEX]}
LAMBDA_LRI=${LRI_VALUES[$LRI_INDEX]}

BASE_DIR="/mnt/shared/scratch/zzhang/hawaii_300"
CODE_DIR="${BASE_DIR}/CAD_VAE_cricket"
PYTHON_SCRIPT="${CODE_DIR}/train.py"

SCAFFOLD_DIR="${BASE_DIR}/scaffold_${SCAFFOLD}"
INPUT_DIR="${SCAFFOLD_DIR}/out_scaffold${SCAFFOLD}_morph_geography"
LD_BLOCK_PKL="${SCAFFOLD_DIR}/scaffold_${SCAFFOLD}_merged_ld_blocks.pkl"

RUN_DIR="${CODE_DIR}/cricket_block_hyper/ablation_y1/${TARGET_COLUMN}"
RUN_DIR="${RUN_DIR}/scaffold_${SCAFFOLD}/block_${BLOCK_ID}/extended"
RUN_DIR="${RUN_DIR}/epochs_${MAX_EPOCHS}/warmup_${BETA_WARMUP_EPOCHS}"
SAVE_DIR="${RUN_DIR}/beta_${BETA}/cls_${LAMBDA_CLS}/tc_${LAMBDA_TC}/cmi_${LAMBDA_CMI}/lri_${LAMBDA_LRI}/seed_${SEED}"

source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${BASE_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p "${SAVE_DIR}"

echo "Started: $(date); scaffold=${SCAFFOLD}; block=${BLOCK_ID}"
echo "beta=${BETA}; CMI=${LAMBDA_CMI}; LRI=${LAMBDA_LRI}; seed=${SEED}"
echo "Output: ${SAVE_DIR}"

python -u "${PYTHON_SCRIPT}" \
    --out_dir "${INPUT_DIR}" \
    --ld_block_pkl "${LD_BLOCK_PKL}" \
    --save_dir "${SAVE_DIR}" \
    --block_id "${BLOCK_ID}" \
    --use_ext_window \
    --target_column "${TARGET_COLUMN}" \
    --num_y_classes 2 \
    --n_splits 5 \
    --inner_val_fraction 0.125 \
    --beta "${BETA}" \
    --beta_warmup_epochs "${BETA_WARMUP_EPOCHS}" \
    --lambda_cls "${LAMBDA_CLS}" \
    --lambda_tc "${LAMBDA_TC}" \
    --lambda_cmi "${LAMBDA_CMI}" \
    --lambda_lri "${LAMBDA_LRI}" \
    --max_epochs "${MAX_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 0 \
    --seed "${SEED}" \
    --device cuda

echo "Finished: $(date)"