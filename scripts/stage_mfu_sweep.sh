#!/bin/bash
set -euo pipefail
NODE=${1:-trn09}
TIME=${2:-00:08:00}
TAG=${3:-mfu_sweep}
CODE_WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium_git
REPO_URL=https://github.com/catslovefish1/AlphaGo_Trainium.git
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
[ -n "$ADDR" ] || { echo "unknown node $NODE" >&2; exit 1; }
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$ADDR" \
  "set -e; if [ -d '$CODE_WS/.git' ]; then git -C '$CODE_WS' fetch --depth=1 origin main; git -C '$CODE_WS' reset --hard origin/main; else git clone --depth=1 --branch main '$REPO_URL' '$CODE_WS'; fi"
JOB=$(ssh -o BatchMode=yes "$ADDR" "cd '$CODE_WS' && sbatch --parsable --partition=compute_fullnode --nodelist='$NODE' --gres=neuron:16 --exclusive --time='$TIME' --job-name='zhiwei_${TAG}' scripts/mfu_sweep.sbatch")
echo "$JOB"
