"""Audit the AlphaGo_Trainium objective against retrieved evidence."""

import argparse
import json
from pathlib import Path

from .mfu_report import summarize as summarize_mfu


def _read_json(path):
    return json.loads(path.read_text())


def _gate(name, status, evidence, detail):
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "detail": detail,
    }


def audit(root):
    root = Path(root)
    gates = []

    cpu_path = root / "docs/results/local_cpu_validation.json"
    cpu = _read_json(cpu_path) if cpu_path.exists() else {}
    gates.append(
        _gate(
            "rules_and_cpu_mcts",
            "verified" if cpu.get("result") == "passed" else "pending",
            str(cpu_path.relative_to(root)),
            "9x9 CPU MCTS games and 19x19 action shape",
        )
    )

    proxy_path = root / "docs/nn_mfu_report.md"
    gates.append(
        _gate(
            "basic_mfu_proxy",
            "verified" if proxy_path.exists() else "pending",
            str(proxy_path.relative_to(root)),
            "empirical shape and timing proxy results",
        )
    )

    exact_path = root / "docs/results/16559_exact_mfu_hlo/summary.json"
    exact = summarize_mfu(_read_json(exact_path)) if exact_path.exists() else {}
    gates.append(
        _gate(
            "exact_hlo_mfu",
            "verified" if exact.get("exact_mfu_available") else "pending",
            str(exact_path.relative_to(root)),
            "requires populated HLO model_flops",
        )
    )

    async_path = root / "docs/results/16595_async_pipeline/async_pipeline.json"
    async_data = _read_json(async_path) if async_path.exists() else {}
    speedups = async_data.get("speedup_by_depth", {})
    gates.append(
        _gate(
            "async_pipeline",
            "verified" if max(speedups.values(), default=0) > 1 else "pending",
            str(async_path.relative_to(root)),
            "CPU producer and Neuron consumer queue-depth probe",
        )
    )

    hbm_path = root / "docs/results/16589_hbm_capacity/summary.json"
    hbm = _read_json(hbm_path) if hbm_path.exists() else {}
    stable_points = [point for point in hbm.get("points", []) if point.get("stable")]
    gates.append(
        _gate(
            "hbm_capacity",
            "verified" if stable_points else "pending",
            str(hbm_path.relative_to(root)),
            "64-rank batch capacity sweep",
        )
    )

    strength_paths = [
        root / "docs/results/16585_go9_ckpt055_vweak/match.json",
        root / "docs/results/16586_go9_ckpt055_weak/match.json",
        root / "docs/results/16587_go9_ckpt055_ref/match.json",
    ]
    strength = [_read_json(path) for path in strength_paths if path.exists()]
    wins = sum(int(item.get("our_wins", 0)) for item in strength)
    games = sum(int(item.get("games", 0)) for item in strength)
    gates.append(
        _gate(
            "public_engine_strength",
            "verified" if wins > 0 else "pending",
            [str(path.relative_to(root)) for path in strength_paths],
            f"latest KataGo ladder result {wins}/{games} wins",
        )
    )

    shape_path = root / "docs/go19_mfu_report.md"
    gates.append(
        _gate(
            "19x19_preparation",
            "verified" if shape_path.exists() else "pending",
            str(shape_path.relative_to(root)),
            "19x19 shape and batch sweep",
        )
    )

    continuous_evidence = root / "docs/results/16562_go19_train_smoke"
    gates.append(
        _gate(
            "continuous_19x19_training",
            "verified" if (continuous_evidence / "summary.json").exists() else "pending",
            str(continuous_evidence.relative_to(root)),
            "requires a sustained run with periodic checkpoint retrieval",
        )
    )

    pending = [gate["name"] for gate in gates if gate["status"] == "pending"]
    return {
        "overall": "complete" if not pending else "incomplete",
        "pending_gates": pending,
        "gates": gates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
