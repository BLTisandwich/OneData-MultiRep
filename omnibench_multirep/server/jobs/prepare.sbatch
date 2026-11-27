#!/bin/bash
#SBATCH --job-name=lhs_llm_prepare
#SBATCH --partition=jobs
#SBATCH --nodelist=hpc-pr-a-pod12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.out
#SBATCH --error=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.err
# usage: sbatch prepare.sbatch   (after download): package data (align/cache/static/qa) (our omni_v1 cache: omnicache.sbatch)
set -euo pipefail
source /scratch/kaist_shh/lhs/llm_intro/code/OneData-MultiRep/omnibench_multirep/server/env.sh
echo "job $SLURM_JOB_ID on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none}, start $(date)"
run_in "$PKG_PATH" "$REPO" python omnibench_multirep/server/make_config.py "$ROOT" "$REPO"
P() { run_in "$PKG_PATH" "$REPO/ppl" python -m omnibench --config "$CONFIG" "$@"; }
P fetch --encoders
P dataset prepare --stage align --workers "$SLURM_CPUS_PER_TASK" | tail -n 3
echo "aligned: $(ls "$ROOT/data/aligned/episodes" | wc -l) episodes $(date)"
P dataset prepare --stage cache | tail -n 2
P dataset prepare --stage static | tail -n 2
P dataset prepare --stage qa | tail -n 30
P dataset check || echo "WARNING: dataset check differs from the frozen snapshot (see above)"
P doctor
echo "done $(date)"
