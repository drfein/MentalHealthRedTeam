#!/bin/bash
# SLURM job that generates conversation turns for preference files.
#SBATCH -J harm_gen_prefs
#SBATCH -o logs/harm_gen_prefs_%j.out
#SBATCH -e logs/harm_gen_prefs_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

export REPO="/sailhome/drfein/hr"
source "${REPO}/slurm/helpers/common_env.sh"
prepare_job_env

mkdir -p logs harm_kl/data/preferences

echo "Generating conversations for existing preference files: $(date)"
python -m src.experiments.harm_kl.generate_preferences

echo "Done: $(date)"
