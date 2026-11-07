#!/bin/bash
#SBATCH --job-name=zhiwei_alphago_match
#SBATCH --partition=compute_fullnode
#SBATCH --nodes=1
#SBATCH --gres=neuron:16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=/home/ralph/zhiwei_wang/logs/%j.out
set -euo pipefail

WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
VENV=/opt/aws_neuronx_venv_pytorch_2_9
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_NUM_CORES=1
export OMP_NUM_THREADS=1
export XLA_USE_BF16=1

cd "$WS"
OUT="$WS/runs/$SLURM_JOB_ID/${TAG:-external_match}"
mkdir -p "$OUT"
python -m alphago.gtp_match \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --opponent-cmd "$OPPONENT_CMD" \
  --games "${GAMES:-20}" \
  --out "$OUT/match.json" \
  --device xla
