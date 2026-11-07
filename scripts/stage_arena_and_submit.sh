#!/bin/bash
set -euo pipefail
NODE=${1:?usage: stage_arena_and_submit.sh <node> <config> <checkpoint> [time] [tag] [opponent]}
CONFIG=${2:?usage: stage_arena_and_submit.sh <node> <config> <checkpoint> [time] [tag] [opponent]}
CHECKPOINT=${3:?usage: stage_arena_and_submit.sh <node> <config> <checkpoint> [time] [tag] [opponent]}
TIME=${4:-00:10:00}
TAG=${5:-arena}
OPPONENT=${6:-random}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ADDR" "mkdir -p '$WS' '$CHECKPOINT_DIR' /home/ralph/zhiwei_wang/logs"
rsync -a --delete --exclude '.git' --exclude 'runs' --exclude 'reports' ./ "$ADDR:$WS/"
rsync -a "$CHECKPOINT" "$ADDR:$CHECKPOINT"
JOB=$(ssh -o BatchMode=yes "$ADDR" "cd '$WS' && sbatch --parsable --partition=compute_fullnode --nodelist='$NODE' --gres=neuron:16 --exclusive --time='$TIME' --job-name='zhiwei_${TAG}' --export=ALL,CONFIG='$CONFIG',CHECKPOINT='$CHECKPOINT',TAG='$TAG',OPPONENT='$OPPONENT' scripts/arena.sbatch")
echo "$JOB"
