#!/bin/bash
#SBATCH -J fairmt_eval
#SBATCH -o logs/fairmt_eval_%j.out
#SBATCH -e logs/fairmt_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

export USE_TF=0

FAIRMT_DIR="/nlp/scr/drfein/fairmt"
DPO_OUT="/nlp/scr/drfein/hr/dpo_hh_token"

if [ ! -d "${FAIRMT_DIR}/FairMT_1K" ]; then
  echo "Cloning FairMT-bench..."
  mkdir -p "${FAIRMT_DIR}"
  git clone --depth 1 https://github.com/FanZT6/FairMT-bench.git "${FAIRMT_DIR}"
fi

echo "=== with_token run ==="
python -m hr.run fairmt generate \
  --model_id "meta-llama/Llama-3.2-1B-Instruct" \
  --embedding_path "${DPO_OUT}/hh_reminder_embedding.pt" \
  --data_dir "${FAIRMT_DIR}/FairMT_1K" \
  --save_dir "${DPO_OUT}/fairmt_results/with_token"

echo "=== baseline run (no token) ==="
python -m hr.run fairmt generate \
  --model_id "meta-llama/Llama-3.2-1B-Instruct" \
  --no_token \
  --data_dir "${FAIRMT_DIR}/FairMT_1K" \
  --save_dir "${DPO_OUT}/fairmt_results/baseline"

echo "Done: $(date)"
