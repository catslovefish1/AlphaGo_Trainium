"""Runtime helpers for torch-neuronx XLA."""

import os


def init_distributed():
    import torch.distributed as dist
    import torch_xla.distributed.xla_backend  # noqa: F401

    if not dist.is_initialized():
        dist.init_process_group("xla")
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def device():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def ordinal():
    import torch_xla.core.xla_model as xm

    return int(xm.get_local_ordinal())


def optimizer_step(optimizer):
    import torch_xla.core.xla_model as xm

    xm.optimizer_step(optimizer, barrier=True)


def mark_step():
    import torch_xla.core.xla_model as xm

    xm.mark_step()


def rendezvous(tag):
    import torch_xla.core.xla_model as xm

    xm.rendezvous(tag)
