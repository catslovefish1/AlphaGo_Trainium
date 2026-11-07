"""Standard-library checks for the CPU/device overlap probe."""

from .pipeline_probe import prepare_batch, run_cpu_probe


def run():
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        print("pipeline tests skipped: numpy is unavailable")
        return
    first = prepare_batch(2, 9, 3)
    second = prepare_batch(2, 9, 3)
    if first.shape != (2, 4, 9, 9):
        raise AssertionError(f"unexpected static batch shape {first.shape}")
    if not (first == second).all():
        raise AssertionError("batch preparation is not deterministic")
    result = run_cpu_probe(3, 2, 9, 0.001)
    if result["shape"] != [2, 4, 9, 9]:
        raise AssertionError("pipeline changed its batch shape")
    if result["pipelined_s"] <= 0 or result["sequential_s"] <= 0:
        raise AssertionError("pipeline timing did not run")
    print("pipeline tests passed")


if __name__ == "__main__":
    run()
