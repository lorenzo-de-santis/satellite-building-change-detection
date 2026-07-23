#!/bin/bash
#SBATCH --job-name=levir_final_figs
#SBATCH --account=3163450
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/levir_project}
SUBMISSION_DIR=${SUBMISSION_DIR:-${PROJECT_DIR}/levir_final/submission}
DATA_ROOT=${DATA_ROOT:-${PROJECT_DIR}/data/LEVIR-CD}

if [ ! -d "${DATA_ROOT}/train/A" ]; then
  DATA_ROOT=${PROJECT_DIR}/data
fi

cd "${SUBMISSION_DIR}"
mkdir -p logs results/figures

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "${PROJECT_DIR}/env"

RUN_DIR=${RUN_DIR:-$(ls -td results/final_run_* | head -1)}
if [ -z "${RUN_DIR}" ] || [ ! -d "${RUN_DIR}" ]; then
  echo "No training run found. Set RUN_DIR=/path/to/final_run_*." >&2
  exit 1
fi

echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : ${SLURMD_NODENAME}"
echo "Start         : $(date)"
echo "Submission dir: ${SUBMISSION_DIR}"
echo "Data root     : ${DATA_ROOT}"
echo "Run dir       : ${RUN_DIR}"

python -u generate_figures.py \
  --data_root "${DATA_ROOT}" \
  --output_dir results/figures \
  --run_name figures_$(date +%Y%m%d_%H%M%S) \
  --stats_json checkpoints/shared_dataset_stats.json \
  --simple_cnn_checkpoint "${RUN_DIR}/simple_cnn/best.pt" \
  --unet_checkpoint "${RUN_DIR}/unet/best.pt" \
  --siamese_temporal_attention_checkpoint "${RUN_DIR}/siamese_temporal_attention_unet/best.pt" \
  --simple_cnn_history "${RUN_DIR}/simple_cnn/history.csv" \
  --unet_history "${RUN_DIR}/unet/history.csv" \
  --siamese_temporal_attention_history "${RUN_DIR}/siamese_temporal_attention_unet/history.csv" \
  --split test \
  --patch_size 256 \
  --batch_size 8 \
  --num_workers 2 \
  --threshold 0.5 \
  --max_eval_patches 0 \
  --min_change_ratio 0.002 \
  --seed 42

echo "End: $(date)"
