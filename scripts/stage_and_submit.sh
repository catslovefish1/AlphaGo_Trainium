#!/bin/bash
set -euo pipefail
NODE=${1:?usage: stage_and_submit.sh <node> <config> [time] [tag] [resume] [async_nrt]}
CONFIG=${2:?usage: stage_and_submit.sh <node> <config> [time] [tag] [resume]}
TIME=${3:-00:30:00}
TAG=${4:-go9_smoke}
RESUME=${5:-}
ASYNC_NRT=${6:-0}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
CODE_WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium_git
REPO_URL=https://github.com/catslovefish1/AlphaGo_Trainium.git
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
mkdir -p "$WS" /home/ralph/zhiwei_wang/logs
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ADDR" "set -e; mkdir -p '$WS' /home/ralph/zhiwei_wang/logs; if [ -d '$CODE_WS/.git' ]; then git -C '$CODE_WS' fetch --depth=1 origin main; git -C '$CODE_WS' reset --hard origin/main; else git clone --depth=1 --branch main '$REPO_URL' '$CODE_WS'; fi"
if [ -n "$RESUME" ]; then
  RESUME_DIR=$(dirname "$RESUME")
  if ! ssh -o BatchMode=yes "$ADDR" "test -f '$RESUME'"; then
    [ -f "$RESUME" ] || { echo "checkpoint missing on node and head01: $RESUME" >&2; exit 1; }
    ssh -o BatchMode=yes "$ADDR" "mkdir -p '$RESUME_DIR'"
    rsync -a "$RESUME" "$ADDR:$RESUME"
  fi
fi
EXPORT="ALL,CONFIG=$CONFIG,TAG=$TAG,ASYNC_NRT=$ASYNC_NRT"
if [ -n "$RESUME" ]; then
  EXPORT="$EXPORT,RESUME=$RESUME"
fi
JOB=$(ssh -o BatchMode=yes "$ADDR" "cd '$CODE_WS' && sbatch --parsable --partition=compute_fullnode --nodelist='$NODE' --gres=neuron:16 --exclusive --time='$TIME' --job-name='zhiwei_${TAG}' --export=$EXPORT scripts/go_loop.sbatch")
echo "$JOB"
