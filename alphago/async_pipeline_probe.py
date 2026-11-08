"""Compare sequential and CPU-producer pipelined Neuron inference."""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from . import runtime
from .model import build_model


def host_batch(batch_size, size, seed):
    return np.random.default_rng(seed).random(
        (batch_size, 4, size, size), dtype=np.float32
    )


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50.0) * 1000.0),
        "p95_ms": float(np.percentile(values, 95.0) * 1000.0),
        "samples": len(values),
    }


def consume(runtime_module, model, batch, device):
    import torch

    inputs = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        outputs = model(inputs)
    runtime_module.mark_step()
    outputs[0].cpu()


def sequential(runtime_module, model, device, steps, batch_size, size, rank):
    samples = []
    for index in range(steps):
        started = time.perf_counter()
        consume(
            runtime_module,
            model,
            host_batch(batch_size, size, rank * 1000 + index),
            device,
        )
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def pipelined(runtime_module, model, device, steps, batch_size, size, rank, depth):
    batches = queue.Queue(maxsize=depth)
    finished = object()

    def producer():
        for index in range(steps):
            batches.put(host_batch(batch_size, size, rank * 1000 + index))
        batches.put(finished)

    thread = threading.Thread(target=producer, name=f"cpu-producer-{depth}")
    started_total = time.perf_counter()
    thread.start()
    samples = []
    while True:
        batch = batches.get()
        if batch is finished:
            break
        started = time.perf_counter()
        consume(runtime_module, model, batch, device)
        samples.append(time.perf_counter() - started)
    thread.join()
    total_per_step = (time.perf_counter() - started_total) / steps
    result = summarize([total_per_step] * steps)
    result["device_consumer"] = summarize(samples)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    device = runtime.device()
    model = build_model(args.size, args.channels, args.blocks).to(device)
    model.eval()
    for index in range(args.warmup):
        consume(runtime, model, host_batch(args.batch_size, args.size, rank + index), device)
    runtime.rendezvous("async-pipeline-warmup")

    sequential_result = sequential(
        runtime, model, device, args.steps, args.batch_size, args.size, rank
    )
    runtime.rendezvous("async-pipeline-sequential")
    depths = {}
    for depth in (1, 2, 4):
        depths[str(depth)] = pipelined(
            runtime, model, device, args.steps, args.batch_size, args.size, rank, depth
        )
        runtime.rendezvous(f"async-pipeline-depth-{depth}")

    if rank == 0:
        result = {
            "world": world,
            "batch_size": args.batch_size,
            "shape": [args.batch_size, 4, args.size, args.size],
            "warmup": args.warmup,
            "steps": args.steps,
            "sequential": sequential_result,
            "pipelined": depths,
            "speedup_by_depth": {
                depth: sequential_result["mean_ms"] / value["mean_ms"]
                for depth, value in depths.items()
                if value["mean_ms"]
            },
            "interpretation": "CPU preparation overlaps the transfer and Neuron consumer path. Device timeline profiling is required for exact overlap attribution.",
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
