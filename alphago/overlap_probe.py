"""Measure host preparation, host-to-device transfer, and collective latency.

The probe is intentionally separate from training. It reports measured timings
and a simple pipeline lower bound so communication work can be compared with
the host input path without claiming a hardware timeline overlap.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50.0) * 1000.0),
        "p95_ms": float(np.percentile(values, 95.0) * 1000.0),
        "samples": len(samples),
    }


def prepare_host_batch(elements, seed):
    started = time.perf_counter()
    generator = np.random.default_rng(seed)
    batch = generator.standard_normal(elements, dtype=np.float32)
    return batch, time.perf_counter() - started


def measure_host_preparation(elements, steps, seed):
    samples = []
    for step in range(steps):
        _, elapsed = prepare_host_batch(elements, seed + step)
        samples.append(elapsed)
    return summarize(samples)


def measure_host_to_device(runtime, elements, warmup, steps, seed):
    import torch

    device = runtime.device()
    for step in range(warmup):
        batch, _ = prepare_host_batch(elements, seed + step)
        tensor = torch.from_numpy(batch).to(device)
        runtime.mark_step()
        tensor.cpu()

    samples = []
    for step in range(steps):
        batch, _ = prepare_host_batch(elements, seed + warmup + step)
        started = time.perf_counter()
        tensor = torch.from_numpy(batch).to(device)
        runtime.mark_step()
        tensor.cpu()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def measure_all_reduce(runtime, elements, warmup, steps):
    import torch
    import torch.distributed as dist

    device = runtime.device()
    for _ in range(warmup):
        tensor = torch.ones(elements, device=device)
        dist.all_reduce(tensor)
        runtime.mark_step()
        tensor.cpu()

    samples = []
    for _ in range(steps):
        tensor = torch.ones(elements, device=device)
        started = time.perf_counter()
        dist.all_reduce(tensor)
        runtime.mark_step()
        tensor.cpu()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.elements <= 0 or args.steps <= 0 or args.warmup < 0:
        raise ValueError("elements and steps must be positive, warmup cannot be negative")

    from . import runtime

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")

    host = measure_host_preparation(args.elements, args.steps, rank)
    transfer = measure_host_to_device(
        runtime, args.elements, args.warmup, args.steps, rank * 1000
    )
    collective = measure_all_reduce(runtime, args.elements, args.warmup, args.steps)
    runtime.rendezvous("overlap-probe-complete")

    if rank == 0:
        sequential_ms = transfer["mean_ms"] + collective["mean_ms"]
        pipeline_lower_bound_ms = max(transfer["mean_ms"], collective["mean_ms"])
        result = {
            "world": world,
            "elements": args.elements,
            "host_prepare": host,
            "host_to_device": transfer,
            "all_reduce": collective,
            "pipeline_proxy": {
                "sequential_mean_ms": sequential_ms,
                "lower_bound_mean_ms": pipeline_lower_bound_ms,
                "headroom_ms": sequential_ms - pipeline_lower_bound_ms,
                "basis": "max(host-to-device, all-reduce) lower bound; validate true overlap with a device timeline",
            },
        }
        output = Path(args.out or os.environ.get("SMOKE_OUT", "runs/overlap_probe"))
        output.mkdir(parents=True, exist_ok=True)
        output.joinpath("overlap_probe.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
