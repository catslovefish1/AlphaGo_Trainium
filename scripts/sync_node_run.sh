#!/bin/bash
set -euo pipefail
NODE=${1:?usage: sync_node_run.sh <node> <jobid> <tag> [destination]}
JOBID=${2:?usage: sync_node_run.sh <node> <jobid> <tag> [destination]}
TAG=${3:?usage: sync_node_run.sh <node> <jobid> <tag> [destination]}
DEST=${4:-/home/ralph/zhiwei_wang/checkpoint_sync/${JOBID}_${TAG}}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
SRC="$WS/runs/$JOBID/$TAG"
mkdir -p "$DEST"
for name in config.json metrics.jsonl summary.json; do
  if ssh -o BatchMode=yes "$ADDR" "test -f '$SRC/$name'"; then
    scp -q -o BatchMode=yes "$ADDR:$SRC/$name" "$DEST/$name"
  fi
done
for name in $(ssh -o BatchMode=yes "$ADDR" "find '$SRC' -maxdepth 1 -type f -name 'checkpoint_*.pt' -printf '%f\\n' | sort"); do
  scp -q -o BatchMode=yes "$ADDR:$SRC/$name" "$DEST/$name"
done
printf '%s\n' "$DEST"
