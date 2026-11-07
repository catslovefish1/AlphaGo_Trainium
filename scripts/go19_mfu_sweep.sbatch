#!/bin/bash
#SBATCH --job-name=zhiwei_go19_mfu
#SBATCH --partition=compute_fullnode
#SBATCH --nodes=1
#SBATCH --gres=neuron:16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/ralph/zhiwei_wang/logs/%A_%a.out
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

CHANNELS=(32 64 64)
BLOCKS=(3 6 6)
BATCHES=(16 16 32)
CHANNEL=${CHANNELS[$SLURM_ARRAY_TASK_ID]}
BLOCK=${BLOCKS[$SLURM_ARRAY_TASK_ID]}
BATCH=${BATCHES[$SLURM_ARRAY_TASK_ID]}
cd "$CODE_WS"
OUT="$WS/runs/$SLURM_ARRAY_JOB_ID/go19_mfu_sweep/channels_${CHANNEL}_blocks_${BLOCK}_batch_${BATCH}"
mkdir -p "$OUT"
torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 \
  -m alphago.mfu_probe --size 19 --channels "$CHANNEL" --blocks "$BLOCK" \
  --batch-size "$BATCH" --steps 2 --out "$OUT"
