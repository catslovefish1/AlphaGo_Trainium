#!/bin/bash
#SBATCH --job-name=zhiwei_exact_mfu
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
export NEURON_RT_NUM_CORES=1
export OMP_NUM_THREADS=1
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export NEURON_RT_INSPECT_ENABLE=1
export NEURON_RT_INSPECT_SYSTEM_PROFILE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE=1
export NEURON_RT_INSPECT_DEVICE_PROFILE_NUM_WARMUP=2

cd "$CODE_WS"
OUT="$WS/runs/$SLURM_JOB_ID/exact_mfu"
PROFILE="$OUT/profile"
mkdir -p "$PROFILE"
export NEURON_RT_INSPECT_OUTPUT_DIR="$PROFILE"
python -m alphago.mfu_profile_workload --size 9 --channels 64 --blocks 6 --batch-size 64 --steps 6 > "$OUT/workload.json"
export PATH=/opt/aws/neuron/bin:"$PATH"
neuron-explorer view -d "$PROFILE" --output-format summary-json --disable-ui > "$OUT/summary.json"
printf '%s\n' "$OUT"
