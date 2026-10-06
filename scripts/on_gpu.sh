#!/bin/bash
# Usage: scripts/on_gpu.sh <gpu-index|all> <command...>
# Runs one step on the allocated node (SLURM_JOBID or $PSC_JOBID), pinned to one GPU.
JOB="${PSC_JOBID:-$SLURM_JOBID}"
GPU="$1"; shift
PIN=""; [ "$GPU" != "all" ] && PIN="export CUDA_VISIBLE_DEVICES=$GPU;"
exec srun --jobid="$JOB" --overlap -N1 -n1 --gpus-per-node=4 bash -c \
  "$PIN source $(dirname "$0")/env.sh; $*"
