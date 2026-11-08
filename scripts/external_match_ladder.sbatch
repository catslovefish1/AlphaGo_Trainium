#!/bin/bash
#SBATCH --job-name=zhiwei_katago_ladder
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
ASSET=/home/ralph/zhiwei_wang/assets/katago-v1.18.1
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_NUM_CORES=1
export OMP_NUM_THREADS=1
export XLA_USE_BF16=1

cd "$CODE_WS"
OUT="$WS/runs/$SLURM_JOB_ID/${TAG:-external_ladder}"
mkdir -p "$OUT"

for visits in ${VISITS_LIST:-1 8 32}; do
  KATAGO_VISITS="$visits" scripts/prepare_katago_node.sh
  opponent_cmd="$ASSET/katago gtp -config $ASSET/go9_match_${visits}.cfg -model $ASSET/kata9x9-b18c384nbt-20231025.bin.gz"
  printf '%s\n' "$opponent_cmd" > "$OUT/opponent_cmd_${visits}.txt"
  python -m alphago.gtp_match \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --opponent-cmd "$opponent_cmd" \
    --games "${GAMES:-4}" \
    --out "$OUT/match_${visits}.json" \
    --device xla
done

printf 'ladder_out=%s\n' "$OUT"
