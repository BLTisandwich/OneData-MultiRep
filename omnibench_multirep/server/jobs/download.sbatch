#!/bin/bash
#SBATCH --job-name=lhs_llm_download
#SBATCH --partition=jobs
#SBATCH --nodelist=hpc-pr-a-pod12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.out
#SBATCH --error=/scratch/kaist_shh/lhs/llm_intro/logs/%x-%j.err
# usage: sbatch download.sbatch <Lee.tar link> <Lazizbek.tar link> <omni_v1_cache.pkl link>   (Drive share links)
set -euo pipefail
source /scratch/kaist_shh/lhs/llm_intro/code/OneData-MultiRep/omnibench_multirep/server/env.sh
echo "job $SLURM_JOB_ID on $(hostname), GPU(s): ${CUDA_VISIBLE_DEVICES:-none}, start $(date)"
D=$ROOT/data/downloads; mkdir -p "$D" "$ROOT/data/raw"
get() {  # get <link> <file>: gdown into the downloads folder (skipped if already there)
  if [ -s "$2" ]; then echo "$2: already downloaded"; return; fi
  run_in "$MR_PATH" "$D" python -m gdown "$1" -O "$2.part"; mv "$2.part" "$2"
}
get "$3" "$D/omni_v1_cache.pkl"            # first: the multirep sweep only needs this one
get "$1" "$D/Lee.tar"
get "$2" "$D/Lazizbek.tar"
for owner in Lee Lazizbek; do
  if [ ! -d "$ROOT/data/raw/$owner/episodes" ]; then tar -xf "$D/$owner.tar" -C "$ROOT/data/raw"; fi
  echo "$owner: $(ls "$ROOT/data/raw/$owner/episodes" | wc -l) episode folders"
done
ls -la "$D"; echo "done $(date)"
