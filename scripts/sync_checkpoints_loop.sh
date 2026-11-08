#!/bin/bash
set -euo pipefail
NODE=${1:?usage: sync_checkpoints_loop.sh <node> <jobid> <tag> [interval] [destination]}
JOBID=${2:?usage: sync_checkpoints_loop.sh <node> <jobid> <tag> [interval] [destination]}
TAG=${3:?usage: sync_checkpoints_loop.sh <node> <jobid> <tag> [interval] [destination]}
INTERVAL=${4:-1800}
DEST=${5:-/home/ralph/zhiwei_wang/checkpoint_sync/${JOBID}_${TAG}}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
SRC="$WS/runs/$JOBID/$TAG"
mkdir -p "$DEST"

sync_once() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" "test -d '$SRC'" || return 0
  for name in config.json metrics.jsonl summary.json; do
    if ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" "test -f '$SRC/$name'"; then
      rsync -a --partial "$ADDR:$SRC/$name" "$DEST/$name"
    fi
  done
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" \
    "find '$SRC' -maxdepth 1 -type f -name 'checkpoint_*.pt' -printf '%f\\n' | sort" \
    | while read -r name; do
        [ -n "$name" ] || continue
        rsync -a --partial "$ADDR:$SRC/$name" "$DEST/$name"
      done
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" "test -f '/home/ralph/zhiwei_wang/logs/$JOBID.out'"; then
    rsync -a --partial "$ADDR:/home/ralph/zhiwei_wang/logs/$JOBID.out" "$DEST/$JOBID.out"
  fi
}

while true; do
  sync_once
  STATE=$(squeue -h -j "$JOBID" -o '%T' 2>/dev/null || true)
  if [ -z "$STATE" ]; then
    ACCOUNTING=$(sacct -n -X -j "$JOBID" -o State 2>/dev/null | tr -d ' ' | head -1 || true)
    case "$ACCOUNTING" in
      COMPLETED|FAILED|TIMEOUT|CANCELLED|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
        break
        ;;
    esac
  fi
  sleep "$INTERVAL"
done
sync_once
printf '%s\n' "$DEST"
