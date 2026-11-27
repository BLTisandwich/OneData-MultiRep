#!/bin/bash
#SBATCH --job-name=lhs_llm_multirep
#SBATCH --partition=jobs
#SBATCH --nodelist=hpc-pr-a-pod12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.out
#SBATCH --error=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.err
# usage: sbatch multirep.sbatch <cache.pkl> <tag> [run_multirep.py options: --limit 512 --epochs 1 --run-set backbone --route-setting ...]
set -euo pipefail
source /scratch/kaist_shh/lhs/llm_intro/code/OneData-MultiRep/omnibench_multirep/server/env.sh
echo "job $SLURM_JOB_ID on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none}, start $(date)"
CACHE="$1"; TAG="$2"; shift 2
run_in "$MR_PATH" "$REPO" python omnibench_multirep/server/run_multirep.py --notebook "$REPO/multirep_robot.ipynb" --cache "$CACHE" \
  --workdir "$ROOT/results/multirep_$TAG" "$@"
echo "done $(date)"
