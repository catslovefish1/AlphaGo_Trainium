"""Measure host, HBM transfer, Neuron forward, and pipelined step timing."""

import argparse
import json
import queue
import threading
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


def host_batch(batch_size, size, seed):
    return np.random.default_rng(seed).random(
        (batch_size, 4, size, size), dtype=np.float32
    )


def reduce_mean(runtime, value, world):
    import torch
    import torch.distributed as dist

    tensor = torch.tensor([value], dtype=torch.float32, device=runtime.device())
    dist.all_reduce(tensor)
    runtime.mark_step()
    return float(tensor.cpu().item()) / world


def measure_host(steps, batch_size, size, rank):
    samples = []
    for index in range(steps):
        started = time.perf_counter()
        host_batch(batch_size, size, rank * 1000 + index)
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def measure_transfer(runtime, steps, warmup, batch_size, size, rank):
    import torch

    device = runtime.device()
    for index in range(warmup):
        batch = host_batch(batch_size, size, rank * 1000 + index)
        tensor = torch.from_numpy(batch).to(device)
        runtime.mark_step()
        tensor.cpu()
    samples = []
    for index in range(steps):
        batch = host_batch(batch_size, size, rank * 1000 + warmup + index)
        started = time.perf_counter()
        tensor = torch.from_numpy(batch).to(device)
        runtime.mark_step()
        tensor.cpu()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def measure_device_to_host(runtime, steps, warmup, batch_size, size, rank):
    import torch

    device = runtime.device()
    tensor = torch.from_numpy(host_batch(batch_size, size, rank)).to(device)
    runtime.mark_step()
    tensor.cpu()
    for _ in range(warmup):
        tensor.cpu()
    samples = []
    for _ in range(steps):
        started = time.perf_counter()
        tensor.cpu()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def measure_forward(runtime, model, steps, warmup, batch_size, size, rank):
    import torch

    device = runtime.device()
    batch = host_batch(batch_size, size, rank)
    inputs = torch.from_numpy(batch).to(device)
    runtime.mark_step()
    inputs.cpu()
    for _ in range(warmup):
        with torch.no_grad():
            outputs = model(inputs)
        runtime.mark_step()
        outputs[0].cpu()
    samples = []
    for _ in range(steps):
        started = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
        runtime.mark_step()
        outputs[0].cpu()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def measure_step(runtime, model, steps, warmup, batch_size, size, rank):
    import torch

    device = runtime.device()
    for index in range(warmup):
        batch = host_batch(batch_size, size, rank * 1000 + index)
        inputs = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        runtime.mark_step()
        outputs[0].cpu()
    samples = []
    for index in range(steps):
        started = time.perf_counter()
        batch = host_batch(batch_size, size, rank * 1000 + warmup + index)
        inputs = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        runtime.mark_step()
        outputs[0].cpu()
        samples.append(time.perf_counter() - started)
    return summarize(samples)


def measure_pipelined(runtime, model, steps, batch_size, size, rank):
    batches = queue.Queue(maxsize=2)
    finished = object()

    def producer():
        for index in range(steps):
            batches.put(host_batch(batch_size, size, rank * 1000 + index))
        batches.put(finished)

    import torch

    device = runtime.device()
    started_total = time.perf_counter()
    thread = threading.Thread(target=producer, name="host-batch-producer")
    thread.start()
    samples = []
    while True:
        batch = batches.get()
        if batch is finished:
            break
        started = time.perf_counter()
        inputs = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        runtime.mark_step()
        outputs[0].cpu()
        samples.append(time.perf_counter() - started)
    thread.join()
    total_per_step = (time.perf_counter() - started_total) / steps
    result = summarize([total_per_step] * steps)
    result["device_consumer"] = summarize(samples)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from . import runtime
    from .model import build_model

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    device = runtime.device()
    model = build_model(args.size, args.channels, args.blocks).to(device)
    model.eval()
    runtime.rendezvous("tensor-overlap-start")

    host = measure_host(args.steps, args.batch_size, args.size, rank)
    runtime.rendezvous("tensor-overlap-host")
    transfer = measure_transfer(
        runtime, args.steps, args.warmup, args.batch_size, args.size, rank
    )
    runtime.rendezvous("tensor-overlap-transfer")
    device_to_host = measure_device_to_host(
        runtime, args.steps, args.warmup, args.batch_size, args.size, rank
    )
    runtime.rendezvous("tensor-overlap-device-to-host")
    forward = measure_forward(
        runtime, model, args.steps, args.warmup, args.batch_size, args.size, rank
    )
    runtime.rendezvous("tensor-overlap-forward")
    step = measure_step(
        runtime, model, args.steps, args.warmup, args.batch_size, args.size, rank
    )
    runtime.rendezvous("tensor-overlap-step")
    pipelined = measure_pipelined(
        runtime, model, args.steps, args.batch_size, args.size, rank
    )
    runtime.rendezvous("tensor-overlap-pipeline")

    if rank == 0:
        result = {
            "world": world,
            "shape": [args.batch_size, 4, args.size, args.size],
            "warmup": args.warmup,
            "steps": args.steps,
            "host_prepare": host,
            "host_to_hbm": transfer,
            "hbm_to_host": device_to_host,
            "tensor_core_forward": forward,
            "sequential_host_hbm_forward": step,
            "cpu_producer_device_consumer": pipelined,
            "pipeline_proxy": {
                "sequential_mean_ms": step["mean_ms"],
                "pipelined_mean_ms": pipelined["mean_ms"],
                "speedup": step["mean_ms"] / pipelined["mean_ms"]
                if pipelined["mean_ms"]
                else 0.0,
                "headroom_ms": step["mean_ms"] - pipelined["mean_ms"],
                "interpretation": "CPU host preparation overlaps the device consumer; use a device timeline for exact HBM and tensor-core overlap attribution",
            },
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
