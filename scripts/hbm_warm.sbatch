#!/bin/bash
#SBATCH --job-name=zhiwei_hbm_warm
#SBATCH --partition=compute_fullnode
#SBATCH --nodes=1
#SBATCH --gres=neuron:16
#SBATCH --exclusive
#SBATCH --time=00:10:00
#SBATCH --array=0-1%1
#SBATCH --output=/home/ralph/zhiwei_wang/logs/%A_%a.out
set -uo pipefail

WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium
CODE_WS=/home/ralph/zhiwei_wang/AlphaGo_Trainium_git
VENV=/opt/aws_neuronx_venv_pytorch_2_9
export PATH="$VENV/bin:/opt/aws/neuron/bin:$PATH"
export PJRT_DEVICE=NEURON
export NEURON_LOGICAL_NC_CONFIG=2
export NEURON_RT_NUM_CORES=64
export OMP_NUM_THREADS=1
export XLA_USE_BF16=1

BATCHES=(256 512)
BATCH=${BATCHES[$SLURM_ARRAY_TASK_ID]}
OUT="$WS/runs/$SLURM_ARRAY_JOB_ID/hbm_warm/batch_${BATCH}"
mkdir -p "$OUT" "$WS/logs"
cd "$CODE_WS"
MONITOR="$OUT/neuron_monitor.jsonl"
neuron-monitor --verbose 1 > "$MONITOR" 2> "$OUT/neuron_monitor.err" &
MONITOR_PID=$!
STARTED=$(date +%s.%N)
set +e
torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 \
  -m alphago.mfu_probe --size 9 --channels 64 --blocks 6 \
  --batch-size "$BATCH" --warmup 2 --steps 5 --out "$OUT"
MFU_STATUS=$?
set -e
ELAPSED=$(python3 - "$STARTED" <<'PY'
import sys, time
print(time.time() - float(sys.argv[1]))
PY
)
kill -INT "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
set +e
torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 \
  -m alphago.overlap_probe --elements "$((BATCH * 4 * 9 * 9))" \
  --warmup 2 --steps 5 --out "$OUT/overlap"
OVERLAP_STATUS=$?
set -e
python3 - "$OUT" "$BATCH" "$MFU_STATUS" "$OVERLAP_STATUS" "$ELAPSED" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
result = {
    "batch_size": int(sys.argv[2]),
    "mfu_exit_code": int(sys.argv[3]),
    "overlap_exit_code": int(sys.argv[4]),
    "wall_time_s": float(sys.argv[5]),
    "stable": int(sys.argv[3]) == 0 and int(sys.argv[4]) == 0,
}
for name, path in (
    ("mfu_probe", out / "mfu_probe.json"),
    ("overlap_probe", out / "overlap" / "overlap_probe.json"),
):
    if path.exists():
        result[name] = json.loads(path.read_text())
peak = 0
monitor = out / "neuron_monitor.jsonl"
if monitor.exists():
    for line in monitor.read_text().splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        for report in data.get("neuron_runtime_data", []):
            used = report.get("report", {}).get("memory_used", {})
            value = used.get("neuron_runtime_used_bytes", {}).get("neuron_device", 0)
            peak = max(peak, int(value or 0))
result["monitor_peak_neuron_device_bytes"] = peak
(out / "batch_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
PY
