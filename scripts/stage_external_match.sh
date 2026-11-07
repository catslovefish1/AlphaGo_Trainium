#!/bin/bash
set -euo pipefail
NODE=${1:?usage: stage_external_match.sh <node> <config> <checkpoint> [time] [tag] [games]}
CONFIG=${2:?usage: stage_external_match.sh <node> <config> <checkpoint> [time] [tag] [games]}
CHECKPOINT=${3:?usage: stage_external_match.sh <node> <config> <checkpoint> [time] [tag] [games]}
TIME=${4:-00:30:00}
TAG=${5:-external_match}
GAMES=${6:-20}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
CODE_WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium_git
REPO_URL=https://github.com/catslovefish1/AlphaGo_Trainium.git
ASSET=/home/ralph/zhiwei_wang/assets/katago-v1.18.1
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ADDR" \
  "set -e; mkdir -p '$WS' '$CHECKPOINT_DIR' '$ASSET' /home/ralph/zhiwei_wang/logs; if [ -d '$CODE_WS/.git' ]; then git -C '$CODE_WS' fetch --depth=1 origin main; git -C '$CODE_WS' reset --hard origin/main; else git clone --depth=1 --branch main '$REPO_URL' '$CODE_WS'; fi"
if ! ssh -o BatchMode=yes "$ADDR" "test -f '$CHECKPOINT'"; then
  [ -f "$CHECKPOINT" ] || { echo "checkpoint missing on node and head01: $CHECKPOINT" >&2; exit 1; }
  rsync -a "$CHECKPOINT" "$ADDR:$CHECKPOINT"
fi
ssh -o BatchMode=yes "$ADDR" "cd '$CODE_WS' && scripts/prepare_katago_node.sh"
CMD_FILE="$ASSET/opponent_cmd.txt"
ssh -o BatchMode=yes "$ADDR" "printf '%s\\n' '$ASSET/katago gtp -config $ASSET/default_gtp.cfg -model $ASSET/kata9x9-b18c384nbt-20231025.bin.gz' > '$CMD_FILE'"
JOB=$(ssh -o BatchMode=yes "$ADDR" "cd '$CODE_WS' && sbatch --parsable --partition=compute_fullnode --nodelist='$NODE' --gres=neuron:16 --exclusive --time='$TIME' --job-name='zhiwei_${TAG}' --export=ALL,CONFIG='$CONFIG',CHECKPOINT='$CHECKPOINT',TAG='$TAG',GAMES='$GAMES',OPPONENT_CMD_FILE='$CMD_FILE' scripts/external_match.sbatch")
echo "$JOB"
