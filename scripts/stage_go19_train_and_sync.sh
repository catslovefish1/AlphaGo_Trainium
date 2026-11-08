#!/bin/bash
set -euo pipefail

NODE=${1:?usage: stage_go19_train_and_sync.sh <node> <config> [time] [tag] [checkpoint] [async_nrt] [interval] [destination]}
CONFIG=${2:?usage: stage_go19_train_and_sync.sh <node> <config> [time] [tag] [checkpoint] [async_nrt] [interval] [destination]}
TIME=${3:-06:00:00}
TAG=${4:-go19_train}
CHECKPOINT=${5:-}
ASYNC_NRT=${6:-0}
INTERVAL=${7:-1800}
DEST=${8:-}

JOB=$(scripts/stage_go19_train.sh \
  "$NODE" "$CONFIG" "$TIME" "$TAG" "$CHECKPOINT" "$ASYNC_NRT")
printf 'job=%s\n' "$JOB"

if [ -n "$DEST" ]; then
  scripts/sync_checkpoints_loop.sh "$NODE" "$JOB" "$TAG" "$INTERVAL" "$DEST"
else
  scripts/sync_checkpoints_loop.sh "$NODE" "$JOB" "$TAG" "$INTERVAL"
fi
