#!/bin/bash
set -euo pipefail

CODE_WS=${ALPHAGO_CODE_WS:-/home/ralph/zhiwei_wang/AlphaGo_Trainium_git}
REPO_URL=${ALPHAGO_REPO_URL:-https://github.com/catslovefish1/AlphaGo_Trainium.git}
if [ -e "$CODE_WS" ] && [ ! -d "$CODE_WS/.git" ]; then
  echo "refusing to overwrite non-git path $CODE_WS" >&2
  exit 1
fi
if [ -d "$CODE_WS/.git" ]; then
  git -C "$CODE_WS" fetch --depth=1 origin main
  git -C "$CODE_WS" reset --hard origin/main
else
  git clone --depth=1 --branch main "$REPO_URL" "$CODE_WS"
fi
git -C "$CODE_WS" rev-parse --short HEAD
