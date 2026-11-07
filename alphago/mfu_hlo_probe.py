"""Small explicit matmul workload used to validate HLO FLOP accounting."""

import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch_xla.core.xla_model as xm

    class MatmulNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(args.features, args.hidden)
            self.second = nn.Linear(args.hidden, args.features)

        def forward(self, inputs):
            return self.second(torch.relu(self.first(inputs)))

    device = xm.xla_device()
    model = MatmulNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.zeros((args.batch_size, args.features), device=device)
    times = []
    for _ in range(args.steps):
        started = time.perf_counter()
        optimizer.zero_grad()
        output = model(inputs)
        loss = output.square().mean()
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        times.append(time.perf_counter() - started)
    print(
        json.dumps(
            {
                "workload": "explicit_matmul",
                "batch_size": args.batch_size,
                "features": args.features,
                "hidden": args.hidden,
                "steps": args.steps,
                "step_times_s": times,
                "steady_step_s": sum(times[2:]) / len(times[2:]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
