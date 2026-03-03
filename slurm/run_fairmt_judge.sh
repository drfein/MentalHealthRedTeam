#!/bin/bash
#SBATCH -J fairmt_judge
#SBATCH -o logs/fairmt_judge_%j.out
#SBATCH -e logs/fairmt_judge_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 0:20:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

DPO_OUT="/nlp/scr/drfein/hr/dpo_hh_token"

python -m hr.run fairmt judge-heuristic \
  --with_token_dir "${DPO_OUT}/fairmt_results/with_token" \
  --baseline_dir "${DPO_OUT}/fairmt_results/baseline" \
  --save_dir "${DPO_OUT}/fairmt_results/judge_compare"

echo "Done: $(date)"
