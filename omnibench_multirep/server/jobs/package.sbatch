#!/bin/bash
#SBATCH --job-name=lhs_llm_package
#SBATCH --partition=jobs
#SBATCH --nodelist=hpc-pr-a-pod12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.out
#SBATCH --error=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.err
# usage: sbatch package.sbatch <server|smoke> <tracks...>   e.g.  sbatch package.sbatch server qa action
set -euo pipefail
source /scratch/kaist_shh/lhs/llm_intro/code/OneData-MultiRep/omnibench_multirep/server/env.sh
echo "job $SLURM_JOB_ID on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none}, start $(date)"
CFGFILE=$CONFIG; [ "$1" = smoke ] && CFGFILE=$SMOKE_CONFIG; shift
run_in "$PKG_PATH" "$REPO/ppl" python -m omnibench --config "$CFGFILE" run --devices 1 --tracks "$@"
echo "done $(date)"
