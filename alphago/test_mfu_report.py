"""Standard-library checks for exact-MFU profile gating."""

from .mfu_report import summarize


def run():
    missing = summarize({"node": {"model_flops": 0, "mfu_hlo_estimated_percent": 0}})
    assert missing["exact_mfu_available"] is False
    assert missing["record_count"] == 1

    present = summarize(
        {
            "node": {
                "model_flops": 123,
                "mfu_hlo_estimated_percent": 4.5,
                "mfu_inst_estimated_percent": 6.0,
            }
        }
    )
    assert present["exact_mfu_available"] is True
    assert present["mean_mfu_hlo_estimated_percent"] == 4.5
    print("mfu report tests passed")


if __name__ == "__main__":
    run()
