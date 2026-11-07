# CPU and Neuron overlap experiment

The current self-play path prepares a small encoded state batch, launches one
fixed-shape network evaluation, waits for the result, and repeats. That gives a
clean baseline for understanding host-side search cost and device launch cost.

`python3 -m alphago.pipeline_probe --cpu-only` measures the same batch staging
with a CPU callback. The producer-thread path prepares the next fixed-shape
batch while the caller runs the callback. It reports sequential time,
pipelined time, and speedup.

On a Neuron node, run the probe through the same 64-rank launcher after the
basic gate passes.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=64 --max_restarts=0 \
  -m alphago.pipeline_probe --batches 8 --batch-size 2 \
  --out runs/$SLURM_JOB_ID/pipeline_probe.json
```

The device probe keeps all model inputs at `[batch, 4, 9, 9]`, so the XLA
compiler sees one evaluation shape. It also times one 1024-element all-reduce
after the overlap measurements. The first comparison to record is the
sequential versus pipelined speedup. A speedup near one means CPU staging is
hidden already or too small to matter. A larger speedup identifies a useful
place to move the producer into the training loop.

This probe does not change training behavior. It is a measurement step before
introducing asynchronous self-play into `train.py`.

For a Slurm run with the standard node environment, use
`scripts/pipeline_probe.sbatch`. The probe writes one small JSON result under
the job run directory and leaves checkpoints untouched.
