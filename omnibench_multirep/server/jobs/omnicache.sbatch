#!/bin/bash
#SBATCH --job-name=lhs_llm_omnicache
#SBATCH --partition=jobs
#SBATCH --nodelist=hpc-pr-a-pod12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.out
#SBATCH --error=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.err
set -euo pipefail
source /scratch/kaist_shh/lhs/llm_intro/code/OneData-MultiRep/omnibench_multirep/server/env.sh
echo "job $SLURM_JOB_ID on $(hostname), start $(date)"
OMNI_ROOTS="$ROOT/data/aligned" OMNI_OUT="$ROOT/data/omni_v1_cache_aligned.pkl" OMNI_PHASE_ROOTS="$ROOT/data/raw" \
  run_in "$MR_PATH" "$REPO" python omnibench_multirep/server/run_notebook.py "$REPO/omnibench_cache.ipynb"
echo "done $(date)"
