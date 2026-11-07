"""Short fixed-shape 64-rank step probe for MFU profiling."""

import json
import os
import time
import argparse
from pathlib import Path


def estimate_training_flops(size, channels, blocks, batch):
    """Estimate three forward-equivalents for a training step."""
    board = size * size
    forward = 2 * board * 4 * channels * 9
    forward += blocks * 2 * (2 * board * channels * channels * 9)
    forward += 2 * board * channels * 4
    forward += 2 * (4 * board) * (board + 1)
    forward += 2 * board * channels * 2
    forward += 2 * (2 * board) * 64
    forward += 2 * 64
    forward += 2 * 64
    return 3 * forward * batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.steps <= 0 or args.warmup < 0:
        raise ValueError("batch-size and steps must be positive, warmup cannot be negative")
    import torch
    import torch_neuronx  # noqa: F401

    from . import runtime
    from .model import build_model
    import torch.distributed as dist

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    device = runtime.device()
    model = build_model(
        size=args.size, channels=args.channels, blocks=args.blocks
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.zeros((args.batch_size, 4, args.size, args.size), device=device)
    target_policy = torch.zeros((args.batch_size, args.size * args.size + 1), device=device)
    target_policy[:, -1] = 1.0
    target_value = torch.zeros(args.batch_size, device=device)
    for _ in range(args.warmup):
        optimizer.zero_grad()
        logits, value = model(x)
        loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
        loss = loss + torch.nn.functional.mse_loss(value, target_value)
        loss.backward()
        runtime.optimizer_step(optimizer)
    started = time.perf_counter()
    optimizer.zero_grad()
    logits, value = model(x)
    loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
    loss = loss + torch.nn.functional.mse_loss(value, target_value)
    loss.backward()
    runtime.optimizer_step(optimizer)
    initial_step_s = time.perf_counter() - started
    times = []
    for _ in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad()
        logits, value = model(x)
        loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
        loss = loss + torch.nn.functional.mse_loss(value, target_value)
        loss.backward()
        runtime.optimizer_step(optimizer)
        times.append(time.perf_counter() - started)
    def mean_across_ranks(value):
        tensor = torch.tensor([value], dtype=torch.float32, device=device)
        dist.all_reduce(tensor)
        runtime.mark_step()
        return float(tensor.cpu().item()) / world

    initial_step_s = mean_across_ranks(initial_step_s)
    steady_step_s = mean_across_ranks(sum(times) / len(times))
    runtime.rendezvous("mfu-complete")
    if rank == 0:
        flops_per_rank = estimate_training_flops(
            args.size, args.channels, args.blocks, args.batch_size
        )
        aggregate_flops = flops_per_rank * world
        peak_bf16_tflops = float(os.environ.get("TRN2_PEAK_BF16_TFLOPS", "10700"))
        achieved_tflops = aggregate_flops / steady_step_s / 1e12
        output = Path(os.environ.get("SMOKE_OUT", "runs/64_rank_smoke"))
        if args.out:
            output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "world": world,
            "size": args.size,
            "channels": args.channels,
            "blocks": args.blocks,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "steps": args.steps,
            "initial_step_s": initial_step_s,
            "steady_step_s": steady_step_s,
            "steady_step_samples_s": times,
            "samples_per_s_aggregate": world * args.batch_size / steady_step_s,
            "estimated_training_flops_per_rank": flops_per_rank,
            "analytic_aggregate_tflops": achieved_tflops,
            "peak_bf16_tflops_reference": peak_bf16_tflops,
            "analytic_utilization_proxy_percent": achieved_tflops / peak_bf16_tflops * 100.0,
            "mfu_basis": "timing proxy with an analytic 3x forward FLOPs estimate; exact MFU requires Neuron Explorer model_flops",
        }
        output.joinpath("mfu_probe.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
