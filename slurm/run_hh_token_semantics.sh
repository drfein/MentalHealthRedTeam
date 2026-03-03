#!/bin/bash
#SBATCH -J hh_token_semantics
#SBATCH -o logs/hh_token_semantics_%j.out
#SBATCH -e logs/hh_token_semantics_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

export USE_TF=0

python -m hr.run token expound-hh-token-delta

echo "Done: $(date)"
