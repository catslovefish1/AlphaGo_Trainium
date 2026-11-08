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

copy_if_present() {
  local name=$1
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" "test -f '$SRC/$name'"; then
    return 0
  fi
  if ! rsync -a --partial \
    -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
    "$ADDR:$SRC/$name" "$DEST/$name"; then
    printf 'warning: could not copy %s\n' "$name" >&2
    return 1
  fi
}

sync_once() {
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" "test -d '$SRC'"; then
    printf 'warning: run directory is unavailable on %s\n' "$ADDR" >&2
    return 1
  fi
  local failed=0
  for name in config.json metrics.jsonl summary.json; do
    copy_if_present "$name" || failed=1
  done
  local checkpoints
  if checkpoints=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" \
    "find '$SRC' -maxdepth 1 -type f -name 'checkpoint_*.pt' -printf '%f\\n' | sort"); then
    while read -r name; do
      [ -n "$name" ] || continue
      copy_if_present "$name" || failed=1
    done <<< "$checkpoints"
  else
    printf 'warning: could not list checkpoints\n' >&2
    failed=1
  fi
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$ADDR" "test -f '/home/ralph/zhiwei_wang/logs/$JOBID.out'"; then
    if ! rsync -a --partial \
      -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
      "$ADDR:/home/ralph/zhiwei_wang/logs/$JOBID.out" "$DEST/$JOBID.out"; then
      printf 'warning: could not copy Slurm log\n' >&2
      failed=1
    fi
  fi
  return "$failed"
}

while true; do
  sync_once || true
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
sync_once || true
printf '%s\n' "$DEST"
