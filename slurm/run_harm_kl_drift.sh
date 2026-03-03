#!/bin/bash
# SLURM job that runs the standard drift computation.
#SBATCH -J harm_kl_drift
#SBATCH -o logs/harm_kl_drift_%j.out
#SBATCH -e logs/harm_kl_drift_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

mkdir -p harm_kl/results/kl_drift harm_kl/data/preferences

echo "Running kl_drift compute: $(date)"
python -m src.experiments.harm_kl.compute_drift
echo "Done: $(date)"
