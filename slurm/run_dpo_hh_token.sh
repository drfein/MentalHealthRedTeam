#!/bin/bash
#SBATCH -J dpo_hh_token
#SBATCH -o logs/dpo_hh_token_%j.out
#SBATCH -e logs/dpo_hh_token_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 2:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

export USE_TF=0
export HR_DPO_CONFIG="${REPO}/configs/dpo_hh_token.yaml"

echo "Starting DPO HH token run: $(date)"
echo "  config: ${HR_DPO_CONFIG}"
echo "  repo:   ${REPO}"

torchrun \
  --nproc_per_node=8 \
  --master_port=29500 \
  -m src.run dpo train-hh-token \
  --config "${HR_DPO_CONFIG}"

echo "Done: $(date)"
