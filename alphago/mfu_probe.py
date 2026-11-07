"""Short fixed-shape 64-rank step probe for MFU profiling."""

import json
import os
import time
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
    import torch
    import torch_neuronx  # noqa: F401

    from . import runtime
    from .model import build_model

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    device = runtime.device()
    model = build_model(size=9, channels=64, blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.zeros((64, 4, 9, 9), device=device)
    target_policy = torch.zeros((64, 82), device=device)
    target_policy[:, -1] = 1.0
    target_value = torch.zeros(64, device=device)
    times = []
    for step in range(4):
        started = time.time()
        optimizer.zero_grad()
        logits, value = model(x)
        loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
        loss = loss + torch.nn.functional.mse_loss(value, target_value)
        loss.backward()
        runtime.optimizer_step(optimizer)
        if step:
            times.append(time.time() - started)
    runtime.rendezvous("mfu-complete")
    if rank == 0:
        step_seconds = sum(times) / len(times)
        flops_per_rank = estimate_training_flops(9, 64, 6, 64)
        aggregate_flops = flops_per_rank * world
        peak_bf16_tflops = float(os.environ.get("TRN2_PEAK_BF16_TFLOPS", "10700"))
        achieved_tflops = aggregate_flops / step_seconds / 1e12
        output = Path(os.environ.get("SMOKE_OUT", "runs/64_rank_smoke"))
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "world": world,
            "steady_step_s": step_seconds,
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
