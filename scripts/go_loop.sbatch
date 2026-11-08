#!/bin/bash
#SBATCH --job-name=zhiwei_alphago_64_smoke
#SBATCH --partition=compute_fullnode
#SBATCH --nodes=1
#SBATCH --gres=neuron:16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/home/ralph/zhiwei_wang/logs/%j.out
set -euo pipefail

WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
CODE_WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium_git
VENV=/opt/aws_neuronx_venv_pytorch_2_9
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_NUM_CORES=64
export OMP_NUM_THREADS=1
export XLA_USE_BF16=1
if [ "${ASYNC_NRT:-0}" = "1" ]; then
  export TORCH_NEURONX_ENABLE_ASYNC_NRT=1
fi

cd "$CODE_WS"
mkdir -p "$WS/runs/$SLURM_JOB_ID" "$WS/logs"
OUT="$WS/runs/$SLURM_JOB_ID/${TAG:-go9_smoke}"
mkdir -p "$OUT"
echo "job=$SLURM_JOB_ID host=$(hostname) world=64 start=$(date -u +%FT%TZ)"
run64() {
  torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 -m "$@"
}
run64 alphago.distributed_smoke
run64 alphago.mfu_probe
RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  RESUME_ARGS=(--resume "$RESUME")
fi
run64 alphago.train --config "$CONFIG" --out "$OUT" "${RESUME_ARGS[@]}"
echo "job=$SLURM_JOB_ID done=$(date -u +%FT%TZ)"
