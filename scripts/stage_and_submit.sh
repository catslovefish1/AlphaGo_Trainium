#!/bin/bash
set -euo pipefail
NODE=${1:?usage: stage_and_submit.sh <node> <config> [time] [tag] [resume]}
CONFIG=${2:?usage: stage_and_submit.sh <node> <config> [time] [tag] [resume]}
TIME=${3:-00:30:00}
TAG=${4:-go9_smoke}
RESUME=${5:-}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
mkdir -p "$WS" /home/ralph/zhiwei_wang/logs
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ADDR" "mkdir -p '$WS' /home/ralph/zhiwei_wang/logs"
rsync -a --delete --exclude '.git' --exclude 'runs' --exclude 'reports' ./ "$ADDR:$WS/"
if [ -n "$RESUME" ]; then
  RESUME_DIR=$(dirname "$RESUME")
  ssh -o BatchMode=yes "$ADDR" "mkdir -p '$RESUME_DIR'"
  rsync -a "$RESUME" "$ADDR:$RESUME"
fi
EXPORT="ALL,CONFIG=$CONFIG,TAG=$TAG"
if [ -n "$RESUME" ]; then
  EXPORT="$EXPORT,RESUME=$RESUME"
fi
JOB=$(ssh -o BatchMode=yes "$ADDR" "cd '$WS' && sbatch --parsable --partition=compute_fullnode --nodelist='$NODE' --gres=neuron:16 --exclusive --time='$TIME' --job-name='zhiwei_${TAG}' --export=$EXPORT scripts/go_loop.sbatch")
echo "$JOB"
