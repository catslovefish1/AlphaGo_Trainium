"""Compile one fixed-shape policy-value forward graph into a native NEFF."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    import torch
    from torch_neuronx import trace
    import torch_xla.core.xla_model as xm

    from .model import build_model

    device = xm.xla_device()
    model = build_model(size=9, channels=64, blocks=6).eval().to(device)
    example = torch.zeros((args.batch, 4, 9, 9), device=device)
    started = time.time()
    traced = trace(
        model,
        example,
        compiler_workdir=args.workdir,
        inline_weights_to_neff=True,
        cpu_backend=False,
    )
    compile_seconds = time.time() - started
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output))
    result = {
        "status": "passed",
        "batch": args.batch,
        "compile_seconds": compile_seconds,
        "artifact": str(output),
        "compiler_workdir": str(args.workdir),
        "backend": "torch_neuronx.trace",
        "fused_graph": "neuron::forward",
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
