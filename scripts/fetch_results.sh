#!/bin/bash
set -euo pipefail
JOB=${1:?usage: fetch_results.sh <jobid> [tag]}
TAG=${2:-go9_smoke}
WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
NODE=$(sacct -j "$JOB" -n -X -o NodeList | head -1 | tr -d ' ')
ADDR=$(scontrol show node "$NODE" | grep -o 'NodeAddr=[^ ]*' | cut -d= -f2)
REMOTE_RUN="$WS/runs/$JOB/$TAG"
RUN="$WS/runs/${JOB}_${TAG}"
mkdir -p "$RUN"
scp -q "$ADDR:/home/ralph/zhiwei_wang/logs/$JOB.out" "$RUN/job.log" || true
rsync -a "$ADDR:$REMOTE_RUN/" "$RUN/" || true
ls -la "$RUN"
