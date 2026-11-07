#!/bin/bash
#SBATCH --job-name=zhiwei_pipeline_probe
#SBATCH --partition=compute_fullnode
#SBATCH --nodes=1
#SBATCH --gres=neuron:16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --output=/home/ralph/zhiwei_wang/logs/%j.out
set -euo pipefail

WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
VENV=/opt/aws_neuronx_venv_pytorch_2_9
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_NUM_CORES=64
export OMP_NUM_THREADS=1

cd "$WS"
OUT="$WS/runs/$SLURM_JOB_ID/pipeline_probe"
mkdir -p "$OUT"
torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 \
  -m alphago.pipeline_probe --batches 8 --batch-size 2 --out "$OUT/pipeline_probe.json"
