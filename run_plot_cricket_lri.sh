#!/bin/bash
#SBATCH --job-name=plot_cricket_cmi
#SBATCH --output=plot_cricket_cmi_%j.out
#SBATCH --error=plot_cricket_cmi_%j.err
#SBATCH --partition=himem
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

BASE_DIR="/mnt/shared/scratch/zzhang/hawaii_300"
CODE_DIR="${BASE_DIR}/CAD_VAE_cricket"
PYTHON_SCRIPT="${CODE_DIR}/plot_cricket_lri.py"
RESULTS_ROOT="${CODE_DIR}/cricket_block_hyper/ablation_y1/venation"
OUTPUT_DIR="${RESULTS_ROOT}/auc_plots/epochs_200_warmup_50_cls_500_tc_0_cmi_500_seed_1024"

source /mnt/apps/users/zzhang/conda/etc/profile.d/conda.sh
conda activate vae

cd "${CODE_DIR}"

echo "Started: $(date)"
echo "Results: ${RESULTS_ROOT}"

python -u "${PYTHON_SCRIPT}" \
    --root "${RESULTS_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --blocks 176 136 \
    --betas 1 \
    --cls 500 \
    --cmi 500 \
    --tc 0 \
    --epochs 200 \
    --warmup 50 \
    --lri_values 0 20 40 60 80 100 500 1000 \
    --seed 1024 \
    --expected_folds 5

echo "Finished: $(date)"