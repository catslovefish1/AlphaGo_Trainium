#!/bin/bash
#SBATCH --job-name=zhiwei_tensor_overlap
#SBATCH --partition=compute_fullnode
#SBATCH --nodes=1
#SBATCH --gres=neuron:16
#SBATCH --exclusive
#SBATCH --time=00:10:00
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

cd "$CODE_WS"
OUT="$WS/runs/$SLURM_JOB_ID/tensor_overlap"
mkdir -p "$OUT"
torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 \
  -m alphago.tensor_overlap_probe --batch-size 4 --steps 4 --warmup 2 \
  --out "$OUT/tensor_overlap.json"
