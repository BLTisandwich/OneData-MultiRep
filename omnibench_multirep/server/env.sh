# Sourced by the sbatch scripts: GIST pod12, Singularity container, project-local packages.
ROOT=/scratch/kaist_shh/lhs/llm_intro
REPO=$ROOT/code/OneData-MultiRep
SIF=/scratch/kaist_shh/chan_dos.sif
CONFIG=$REPO/ppl/configs/server_qwen25_05b.yaml          # written by omnibench_multirep/server/make_config.py
SMOKE_CONFIG=$REPO/ppl/configs/server_qwen25_05b_smoke.yaml
PKG_PATH=$ROOT/pylib_pkg:$REPO/ppl                         # omnibench package: transformers 5.3, peft, pybullet, ...
MR_PATH=$ROOT/pylib_mr                                     # multirep_robot notebook: container transformers 4.57 + pybullet
export SINGULARITYENV_LD_PRELOAD=/.singularity.d/libs/libcuda.so.1
export SINGULARITYENV_OMP_NUM_THREADS=4
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SINGULARITYENV_TRITON_CACHE_DIR=$ROOT/triton_cache
export SINGULARITYENV_HF_HOME=$ROOT/hf_cache
export SINGULARITYENV_MPLCONFIGDIR=$ROOT/mpl_cache
export SINGULARITYENV_PYTHONUNBUFFERED=1
mkdir -p "$ROOT"/{logs,results,data,assets,mpl_cache}

run_in() {  # run_in <PYTHONPATH> <workdir> <command...>
  local lib="$1" wd="$2"; shift 2
  ${SLURM_JOB_ID:+srun --ntasks=1} env SINGULARITYENV_PYTHONPATH="$lib" singularity exec --nv --bind "$ROOT:$ROOT" --bind /tmp:/tmp --pwd "$wd" "$SIF" "$@"
}
