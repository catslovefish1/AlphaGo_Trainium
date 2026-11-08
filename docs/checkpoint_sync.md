# Checkpoint collection

Run `scripts/sync_checkpoints_loop.sh` on head01 after submitting a node job.

```bash
scripts/sync_checkpoints_loop.sh trn09 <jobid> <tag> 1800
```

The loop copies `config.json`, `metrics.jsonl`, `summary.json`, every checkpoint file, and the Slurm log about every 30 minutes. It performs a final collection after Slurm reports a terminal state. Model weights stay under `checkpoint_sync/` on head01 and are not committed to GitHub.

Individual copy failures are reported and retried on the next interval. A
partial collection remains usable when the node disappears before the final
sync.
