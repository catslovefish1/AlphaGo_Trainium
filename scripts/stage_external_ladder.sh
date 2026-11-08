#!/bin/bash
set -euo pipefail
NODE=${1:?usage: stage_external_ladder.sh <node> <config> <checkpoint> [time] [tag] [games] [visits]}
CONFIG=${2:?usage: stage_external_ladder.sh <node> <config> <checkpoint> [time] [tag] [games] [visits]}
CHECKPOINT=${3:?usage: stage_external_ladder.sh <node> <config> <checkpoint> [time] [tag] [games] [visits]}
TIME=${4:-00:30:00}
TAG=${5:-external_ladder}
GAMES=${6:-4}
VISITS_LIST=${7:-"1 8 32"}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
CODE_WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium_git
REPO_URL=https://github.com/catslovefish1/AlphaGo_Trainium.git
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ADDR" \
  "set -e; mkdir -p '$WS' '$CHECKPOINT_DIR' /home/ralph/zhiwei_wang/logs; if [ -d '$CODE_WS/.git' ]; then git -C '$CODE_WS' fetch --depth=1 origin main; git -C '$CODE_WS' reset --hard origin/main; else git clone --depth=1 --branch main '$REPO_URL' '$CODE_WS'; fi"
if ! ssh -o BatchMode=yes "$ADDR" "test -f '$CHECKPOINT'"; then
  [ -f "$CHECKPOINT" ] || { echo "checkpoint missing on node and head01: $CHECKPOINT" >&2; exit 1; }
  ssh -o BatchMode=yes "$ADDR" "mkdir -p '$CHECKPOINT_DIR'"
  rsync -a "$CHECKPOINT" "$ADDR:$CHECKPOINT"
fi
EXPORT="ALL,CONFIG=$CONFIG,CHECKPOINT=$CHECKPOINT,TAG=$TAG,GAMES=$GAMES,VISITS_LIST='$VISITS_LIST'"
JOB=$(ssh -o BatchMode=yes "$ADDR" \
  "cd '$CODE_WS' && sbatch --parsable --partition=compute_fullnode --nodelist='$NODE' --gres=neuron:16 --exclusive --time='$TIME' --job-name='zhiwei_${TAG}' --export=$EXPORT scripts/external_match_ladder.sbatch")
echo "$JOB"
