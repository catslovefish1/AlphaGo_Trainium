# AlphaGo_Trainium

A short, full-node Trainium2 smoke test for AlphaZero-style learning on 9x9 Go.
The model starts from random weights and trains only on self-play trajectories.

## Project layout

- `alphago/` contains the rules, MCTS, model, distributed runtime, training loop, GTP adapter, and probes.
- `configs/` contains the 9x9 smoke configuration.
- `scripts/` contains the head01 staging, Slurm, and result-fetch helpers.
- `assets/` is reserved for evaluation-only public engines and weights.
- `runs/` and `reports/` hold retrieved results.

## Local checks

```bash
python3 -m alphago.test_core
```

The local core test requires only the Python standard library. Torch checks run on head01 in the Neuron environment.

## Full-node smoke run

The head01 launcher stages the checkout to `trn09`, requests `compute_fullnode`, and starts exactly 64 workers.

```bash
scripts/stage_and_submit.sh trn09 configs/go9_smoke.json 00:30:00 go9_smoke
scripts/fetch_results.sh <jobid> go9_smoke
```

The job uses `NEURON_LOGICAL_NC_CONFIG=2`, `NEURON_RT_NUM_CORES=64`, and the installed XLA backend. It stops after a short training budget. Rank 0 writes checkpoints and aggregated metrics.

For a longer learning run, use `configs/go9_learning_v1.json`. Its checkpoints keep model and optimizer state on rank 0 and regenerate replay after resume, which keeps checkpoint traffic bounded as the buffer grows.

The first larger-batch trial is retained as a diagnostic. `configs/go9_learning_v1b.json` keeps all 64 ranks and the resumable checkpoint mode while using the batch size that passed the original smoke run.

## Native fused-NEFF probe

The optional native probe traces one fixed-shape policy-value forward graph with `torch_neuronx.trace`. The trace API emits a fused `neuron::forward` graph that can be saved as a TorchScript artifact. Run it as a separate short Slurm job when a node window is available.

```bash
sbatch -p compute_fullnode -w trn09 --time=00:10:00 -J zhiwei_native_neff scripts/native_neff_probe.sbatch
```

The XLA training path remains the reference path until this probe produces an artifact and a compile time that fit the node window.

The current native compile and warmed-up overlap results are summarized in
[`docs/parallelism_report.md`](docs/parallelism_report.md).

The next 64-rank learning result is summarized in
[`docs/learning_v1c_report.md`](docs/learning_v1c_report.md).

## Host and collective timing probe

After a smoke allocation, run `python -m alphago.overlap_probe --out runs/overlap_probe` to measure host batch preparation, host-to-device transfer, and 64-rank all-reduce latency. The JSON report also records a pipeline lower bound derived from the measured transfer and collective times. It is a timing proxy that needs a device timeline for a true overlap claim.

## Evaluation boundary

Public engine binaries and weights belong under `assets/` and are used only after a checkpoint exists. They never enter the self-play replay buffer. The GTP adapter accepts standard `boardsize`, `clear_board`, `komi`, `play`, `genmove`, `undo`, `final_score`, and `quit` commands.

## Learning interpretation

The smoke run is a learning-system test rather than a strength claim. It checks whether random initialization, self-play targets, policy/value updates, and distributed synchronization produce measurable movement in a short window. Human learning is a later comparison point for curriculum, memory, exploration, and transfer. Human game records stay outside the training replay until a separate supervised or imitation experiment is defined.

## Prior baseline

The retired prototype completed two 9x9 runs on a single debug node. The first completed 1,234 games but collapsed to early passing. The second used an early-pass guard and reached 12/50 wins against random. The new smoke run checks whether all 64 logical cores can produce a synchronized learning curve.

