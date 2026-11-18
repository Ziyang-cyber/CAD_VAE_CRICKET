#!/bin/bash
#SBATCH --job-name=cadvae_all_blocks
#SBATCH --output=/mnt/shared/scratch/zzhang/hawaii_300/CAD_VAE_cricket/cricket_block_hyper/cadvae_all_blocks_cls500_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/zzhang/hawaii_300/CAD_VAE_cricket/cricket_block_hyper/cadvae_all_blocks_cls500_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-7
#SBATCH --time=64:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16

set -euo pipefail

# Configuration
WORKER_ID=${SLURM_ARRAY_TASK_ID}
N_WORKERS=8

BETA=1
CMI=500
LAMBDA_CLS=500
LAMBDA_TC=500
LAMBDA_LRI=500

MAX_EPOCHS=200
BETA_WARMUP_EPOCHS=50
BATCH_SIZE=32
SEED=1024
TARGET_COLUMN=venation

BASE_DIR="/mnt/shared/scratch/zzhang/hawaii_300"
CODE_DIR="${BASE_DIR}/CAD_VAE_cricket"
PYTHON_SCRIPT="${CODE_DIR}/train.py"

# Environment
source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

cd "${CODE_DIR}"

export PYTHONPATH="${CODE_DIR}:${BASE_DIR}:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"

echo "Started: $(date); worker=${WORKER_ID}"
echo "beta=${BETA}; CMI=${CMI}; TC=${LAMBDA_TC}; LRI=${LAMBDA_LRI}"

# Visit scaffolds in order.
for SCAFFOLD in {1..14}; do

    SCAFFOLD_DIR="${BASE_DIR}/scaffold_${SCAFFOLD}"
    INPUT_DIR="${SCAFFOLD_DIR}/out_scaffold${SCAFFOLD}_morph_geography"
    LD_BLOCK_PKL="${SCAFFOLD_DIR}/scaffold_${SCAFFOLD}_merged_ld_blocks.pkl"

    N_BLOCKS=$(python -c '
import pickle
import sys

with open(sys.argv[1], "rb") as handle:
    print(len(pickle.load(handle)["merged_blocks"]))
' "${LD_BLOCK_PKL}")

    echo "Worker=${WORKER_ID}; scaffold=${SCAFFOLD}; blocks=${N_BLOCKS}"

    # Worker 0 handles blocks 0, 8, 16, ...
    # Worker 1 handles blocks 1, 9, 17, ...
    for ((BLOCK_ID=WORKER_ID; BLOCK_ID<N_BLOCKS; BLOCK_ID+=N_WORKERS)); do

        RUN_DIR="${SCAFFOLD_DIR}/outputs_cadvae_scaffold${SCAFFOLD}_extended_${TARGET_COLUMN}"
        RUN_DIR="${RUN_DIR}_beta${BETA}_cls${LAMBDA_CLS}_tc${LAMBDA_TC}_cmi${CMI}_lri${LAMBDA_LRI}"
        RUN_DIR="${RUN_DIR}_warmup${BETA_WARMUP_EPOCHS}_epochs${MAX_EPOCHS}_batch${BATCH_SIZE}_seed${SEED}_allblocks"
        SAVE_DIR="${RUN_DIR}/block_${BLOCK_ID}"

        # Skip completed blocks.
        if [[ -s "${SAVE_DIR}/vae_test_metrics_all_folds.csv" &&
              -s "${SAVE_DIR}/downstream_summary.csv" ]]; then
            echo "[Skip] Scaffold=${SCAFFOLD}; block=${BLOCK_ID}"
            continue
        fi

        mkdir -p "${SAVE_DIR}"

        echo "Started: $(date); scaffold=${SCAFFOLD}; block=${BLOCK_ID}"
        echo "Output: ${SAVE_DIR}"

        # Train all five outer folds for this block.
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
            --lambda_cmi "${CMI}" \
            --lambda_lri "${LAMBDA_LRI}" \
            --max_epochs "${MAX_EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers 0 \
            --seed "${SEED}" \
            --device cuda

        echo "Finished block: $(date); scaffold=${SCAFFOLD}; block=${BLOCK_ID}"

    done
done

echo "Worker ${WORKER_ID} completed: $(date)"