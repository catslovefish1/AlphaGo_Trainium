"""64-rank XLA startup, device identity, all-reduce, and model-step gate."""

import json
import os
from pathlib import Path


def main():
    import torch
    import torch.distributed as dist
    import torch_neuronx  # noqa: F401

    from . import runtime
    from .model import build_model

    rank, world = runtime.init_distributed()
    if world != 64:
        raise RuntimeError(f"expected WORLD_SIZE=64, got {world}")
    device = runtime.device()
    ordinal = runtime.ordinal()
    marker = torch.tensor([float(ordinal)], device=device)
    total = marker.clone()
    dist.all_reduce(total)
    runtime.mark_step()
    observed = float(total.cpu().item())
    expected = sum(range(64))
    if observed != expected:
        raise RuntimeError(f"all-reduce mismatch {observed} != {expected}")

    model = build_model(size=9, channels=64, blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.zeros((2, 4, 9, 9), device=device)
    target_policy = torch.zeros((2, 82), device=device)
    target_policy[:, -1] = 1.0
    target_value = torch.zeros(2, device=device)
    logits, value = model(x)
    loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
    loss = loss + torch.nn.functional.mse_loss(value, target_value)
    loss.backward()
    runtime.optimizer_step(optimizer)
    runtime.rendezvous("smoke-complete")

    if rank == 0:
        output = Path(os.environ.get("SMOKE_OUT", "runs/64_rank_smoke"))
        output.mkdir(parents=True, exist_ok=True)
        output.joinpath("distributed_smoke.json").write_text(
            json.dumps({"world": world, "ordinal_sum": observed, "status": "passed"}, indent=2)
        )
        print(json.dumps({"world": world, "ordinal_sum": observed, "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()
