"""Normalize Neuron Explorer summaries for exact-MFU gating."""

import argparse
import json
from pathlib import Path


_METRIC_KEYS = (
    "model_flops",
    "mfu_hlo_estimated_percent",
    "mfu_inst_estimated_percent",
    "hfu_estimated_percent",
    "mbu_estimated_percent",
    "hbm_read_bytes",
    "hbm_write_bytes",
    "dma_active_time",
    "dma_transfer_time",
    "tensor_engine_active_time",
    "total_active_time",
    "total_exec_time",
    "total_time",
)


def _records(value, path=()):
    if isinstance(value, dict):
        if "model_flops" in value:
            record = {"path": ".".join(path)}
            for key in _METRIC_KEYS:
                if key in value:
                    record[key] = value[key]
            yield record
        for key, child in value.items():
            yield from _records(child, path + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _records(child, path + (str(index),))


def summarize(payload):
    records = list(_records(payload))
    exact_records = [
        record
        for record in records
        if float(record.get("model_flops", 0) or 0) > 0
    ]
    hlo_values = [
        float(record["mfu_hlo_estimated_percent"])
        for record in exact_records
        if record.get("mfu_hlo_estimated_percent") is not None
    ]
    instruction_values = [
        float(record["mfu_inst_estimated_percent"])
        for record in records
        if record.get("mfu_inst_estimated_percent") is not None
    ]
    return {
        "exact_mfu_available": bool(exact_records and hlo_values),
        "record_count": len(records),
        "model_flops_record_count": len(exact_records),
        "mean_mfu_hlo_estimated_percent": (
            sum(hlo_values) / len(hlo_values) if hlo_values else None
        ),
        "mean_mfu_inst_estimated_percent": (
            sum(instruction_values) / len(instruction_values)
            if instruction_values
            else None
        ),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.summary.read_text()))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
