#!/bin/bash
#SBATCH --job-name=lhs_llm_demo
#SBATCH --partition=jobs
#SBATCH --nodelist=hpc-pr-a-pod12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.out
#SBATCH --error=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.err
# usage: sbatch [--gres=gpu:1] demo.sbatch <sweep workdir> <cache.pkl> <episodes root> [omni_demo.py options]   (CPU without --gres)
set -euo pipefail
source /scratch/kaist_shh/lhs/llm_intro/code/OneData-MultiRep/omnibench_multirep/server/env.sh
echo "job $SLURM_JOB_ID on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none}, start $(date)"
WD="$1"; CACHE="$2"; EPR="$3"; shift 3
export OMNI_PHASE_ROOTS="$ROOT/data/raw"          # expert phases of aligned episodes come from the raw episodes
run_in "$MR_PATH" "$REPO" python omnibench_multirep/server/omni_demo.py --workdir "$WD" --cache "$CACHE" --episodes-root "$EPR" "$@"
echo "done $(date)"
