#!/bin/bash
set -eo pipefail

prepare_job_env() {
  local repo_default="/sailhome/drfein/hr"
  export REPO="${REPO:-$repo_default}"
  mkdir -p "$REPO/logs"
  cd "$REPO"

  set +u
  source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
  conda activate saerm
  set -u

  export HF_HOME="${HF_HOME:-/nlp/scr/drfein/hf_home}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/nlp/scr/drfein/hf_home}"
}
