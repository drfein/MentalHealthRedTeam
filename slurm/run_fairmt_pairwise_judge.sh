#!/bin/bash
#SBATCH -J fairmt_pairwise
#SBATCH -o logs/fairmt_pairwise_%j.out
#SBATCH -e logs/fairmt_pairwise_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 2:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  exit 1
fi

DPO_OUT="/nlp/scr/drfein/hr/dpo_hh_token"

python -m src.run fairmt judge-pairwise-llm \
  --model "gpt-4o-mini" \
  --with_token_dir "${DPO_OUT}/fairmt_results/with_token" \
  --baseline_dir "${DPO_OUT}/fairmt_results/baseline" \
  --save_dir "${DPO_OUT}/fairmt_results/judge_pairwise_llm"

echo "Done: $(date)"
