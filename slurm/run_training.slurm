#!/bin/bash
#SBATCH --job-name=levir_final_train
#SBATCH --account=3163450
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$HOME/levir_project}
SUBMISSION_DIR=${SUBMISSION_DIR:-${PROJECT_DIR}/levir_final/submission}
DATA_ROOT=${DATA_ROOT:-${PROJECT_DIR}/data/LEVIR-CD}

if [ ! -d "${DATA_ROOT}/train/A" ]; then
  DATA_ROOT=${PROJECT_DIR}/data
fi

cd "${SUBMISSION_DIR}"
mkdir -p logs results

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate "${PROJECT_DIR}/env"

echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : ${SLURMD_NODENAME}"
echo "Start         : $(date)"
echo "Submission dir: ${SUBMISSION_DIR}"
echo "Data root     : ${DATA_ROOT}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu  :", torch.cuda.get_device_name(0))
PY

python -u run_training.py \
  --data_root "${DATA_ROOT}" \
  --output_dir results \
  --config configs/final_models.json \
  --stats_json checkpoints/shared_dataset_stats.json \
  --device auto \
  --patch_size 256 \
  --num_workers 8 \
  --seed 42

echo "End: $(date)"
