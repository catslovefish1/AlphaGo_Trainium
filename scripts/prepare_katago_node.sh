#!/bin/bash
set -euo pipefail

ASSET=/home/ralph/zhiwei_wang/assets/katago-v1.18.1
ZIP="$ASSET/katago-v1.18.1-eigenavx2-linux-x64.zip"
MODEL="$ASSET/kata9x9-b18c384nbt-20231025.bin.gz"
VISITS=${KATAGO_VISITS:-32}
MATCH_CONFIG="$ASSET/go9_match_${VISITS}.cfg"
mkdir -p "$ASSET"
if [ ! -x "$ASSET/katago" ]; then
  curl -L --fail --silent --show-error \
    https://github.com/lightvector/KataGo/releases/download/v1.18.1/katago-v1.18.1-eigenavx2-linux-x64.zip \
    -o "$ZIP"
  echo "33e79780dbe3bf6ee859e16f64952cdfc90f7210c8f71ad978ffcba85ad20d79  $ZIP" | sha256sum -c -
  python3 - "$ZIP" "$ASSET" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
  chmod +x "$ASSET/katago"
fi
if [ ! -f "$MODEL" ]; then
  curl -L --fail --silent --show-error \
    https://media.katagotraining.org/uploaded/networks/models_extra/kata9x9-b18c384nbt-20231025.bin.gz \
    -o "$MODEL"
fi
test -x "$ASSET/katago"
test -s "$MODEL"
if [ ! -f "$MATCH_CONFIG" ]; then
  cp "$ASSET/default_gtp.cfg" "$MATCH_CONFIG"
  sed -i "s/^maxVisits = .*/maxVisits = $VISITS/" "$MATCH_CONFIG"
  sed -i 's/^numSearchThreads = .*/numSearchThreads = 1/' "$MATCH_CONFIG"
fi
test -s "$MATCH_CONFIG"
echo "KataGo ready at $ASSET"
