"""Single-rank workload used for Neuron Explorer exact MFU capture."""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    import torch
    import torch_xla.core.xla_model as xm

    from .model import build_model

    device = xm.xla_device()
    model = build_model(args.size, args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.zeros(
        (args.batch_size, 4, args.size, args.size), device=device
    )
    target_policy = torch.zeros(
        (args.batch_size, args.size * args.size + 1), device=device
    )
    target_policy[:, -1] = 1.0
    target_value = torch.zeros(args.batch_size, device=device)
    times = []
    for _ in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad()
        logits, value = model(inputs)
        loss = -(target_policy * torch.log_softmax(logits, dim=1)).sum(1).mean()
        loss = loss + torch.nn.functional.mse_loss(value, target_value)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        times.append(time.perf_counter() - started)
    print(
        json.dumps(
            {
                "size": args.size,
                "channels": args.channels,
                "blocks": args.blocks,
                "batch_size": args.batch_size,
                "steps": args.steps,
                "step_times_s": times,
                "steady_step_s": sum(times[2:]) / len(times[2:]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
