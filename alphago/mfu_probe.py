"""Short fixed-shape 64-rank step probe for MFU profiling."""

import json
import os
import time
from pathlib import Path


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
        output = Path(os.environ.get("SMOKE_OUT", "runs/64_rank_smoke"))
        output.mkdir(parents=True, exist_ok=True)
        output.joinpath("mfu_probe.json").write_text(
            json.dumps({"world": world, "steady_step_s": sum(times) / len(times)}, indent=2)
        )
        print(json.dumps({"world": world, "steady_step_s": sum(times) / len(times)}), flush=True)


if __name__ == "__main__":
    main()
