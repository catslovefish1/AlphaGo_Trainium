"""Measure CPU staging, static-shape device work, and collective latency.

The probe keeps device execution on the caller thread. A producer thread only
prepares NumPy batches, which makes the overlap experiment easy to compare
with the existing synchronous self-play path.
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

def prepare_batch(batch_size, size, seed):
    """Create a fixed-shape batch that stands in for encoded self-play states."""
    import numpy as np

    rng = np.random.default_rng(seed)
    batch = rng.random((batch_size, 4, size, size), dtype=np.float32)
    return batch


def sequential_batches(batch_count, batch_size, size, work):
    """Prepare batches and run a CPU callback without overlap."""
    started = time.perf_counter()
    for index in range(batch_count):
        batch = prepare_batch(batch_size, size, index)
        work(batch)
    return time.perf_counter() - started


def pipelined_batches(batch_count, batch_size, size, work, queue_depth=2):
    """Prepare the next batch on a CPU thread while the caller runs ``work``."""
    batches = queue.Queue(maxsize=queue_depth)
    finished = object()

    def producer():
        try:
            for index in range(batch_count):
                batches.put(prepare_batch(batch_size, size, index))
            batches.put(finished)
        except BaseException as error:  # propagate worker failures to the caller
            batches.put(error)

    thread = threading.Thread(target=producer, name="cpu-batch-producer")
    thread.start()
    started = time.perf_counter()
    while True:
        batch = batches.get()
        if batch is finished:
            break
        if isinstance(batch, BaseException):
            raise batch
        work(batch)
    thread.join()
    return time.perf_counter() - started


def run_cpu_probe(batch_count, batch_size, size, cpu_work_s):
    """Return a deterministic overlap comparison without requiring Torch."""

    def work(_batch):
        if cpu_work_s:
            time.sleep(cpu_work_s)

    sequential_s = sequential_batches(batch_count, batch_size, size, work)
    pipelined_s = pipelined_batches(batch_count, batch_size, size, work)
    return {
        "batch_count": batch_count,
        "batch_size": batch_size,
        "shape": [batch_size, 4, size, size],
        "sequential_s": sequential_s,
        "pipelined_s": pipelined_s,
        "speedup": sequential_s / pipelined_s if pipelined_s else 0.0,
    }


def run_neuron_probe(batch_count, batch_size, size, channels, blocks):
    """Run the overlap comparison with a fixed-shape XLA forward callback."""
    import torch

    from . import runtime
    from .model import build_model

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    device = runtime.device()
    model = build_model(size=size, channels=channels, blocks=blocks).to(device)

    def work(batch):
        inputs = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        runtime.mark_step()
        outputs[0].cpu()

    work(prepare_batch(batch_size, size, 0))
    runtime.rendezvous("pipeline-warmup")

    sequential_s = sequential_batches(batch_count, batch_size, size, work)
    runtime.rendezvous("pipeline-sequential")
    pipelined_s = pipelined_batches(batch_count, batch_size, size, work)
    runtime.rendezvous("pipeline-pipelined")

    import torch.distributed as dist

    for _ in range(2):
        collective = torch.ones(1024, device=device)
        dist.all_reduce(collective)
        runtime.mark_step()
        collective.cpu()
    dist.barrier()
    runtime.mark_step()
    started = time.perf_counter()
    collective = torch.ones(1024, device=device)
    dist.all_reduce(collective)
    runtime.mark_step()
    collective_s = time.perf_counter() - started
    if rank == 0:
        result = {
            "world": world,
            "batch_count": batch_count,
            "batch_size": batch_size,
            "shape": [batch_size, 4, size, size],
            "sequential_s": sequential_s,
            "pipelined_s": pipelined_s,
            "speedup": sequential_s / pipelined_s if pipelined_s else 0.0,
            "all_reduce_1024_float32_s": collective_s,
        }
        print(json.dumps(result), flush=True)
        return result
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--cpu-work-ms", type=float, default=2.0)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.cpu_only:
        result = run_cpu_probe(
            args.batches,
            args.batch_size,
            args.size,
            args.cpu_work_ms / 1000.0,
        )
    else:
        result = run_neuron_probe(
            args.batches,
            args.batch_size,
            args.size,
            args.channels,
            args.blocks,
        )
    if result is not None and args.cpu_only:
        print(json.dumps(result), flush=True)
    if result is not None and args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
