# AlphaGo_Trainium current status

Updated 2026-11-08 after the trn09 booking ended.

## Allocation

The trn09 instance is no longer available at the infrastructure level. No new
Slurm job was submitted after the booking ended. The remaining work is staged
for the next allocation and does not claim that an unrun experiment completed.

## Verified results

- The 64-rank 9x9 path initialized, synchronized, trained, and wrote
  checkpoints through checkpoint 055.
- Policy and value losses moved downward during the from-scratch and tuned
  continuations. The strongest recorded short continuation reached policy loss
  near 2.48 at checkpoint 055.
- The public KataGo ladder at checkpoints 045, 050, and 055 recorded 0/4 wins
  at 1, 8, and 32 visits. The model has not yet demonstrated a win against the
  very-weak public reference.
- The 64-rank HBM capacity sweep was stable through per-rank batch 8,192.
  Per-rank batch 2,048 was the best measured warmed throughput point at
  3,646,351 samples/s. Batch 4,096 remained stable at 3,539,827 samples/s.
- The CPU producer and Neuron consumer queue probe reached a 1.154x speedup at
  queue depth 4. Queue depth 2 reached 1.146x and is the smaller practical
  buffer for the next training comparison.
- The 19x19 shape sweep completed, and a short 19x19 smoke run reached early
  training iterations with decreasing losses. Continuous 19x19 training was
  not started because the exact-MFU gate was unresolved.

## MFU status

The native fused-NEFF probe compiled successfully. Neuron Explorer captured
device and HLO artifacts, but the installed SDK combination reported
`model_flops: 0` for both the policy-value profile and the explicit matmul
validation. The repository therefore retains instruction-level MFU, HBM, DMA,
tensor-engine, and communication measurements without claiming an exact HLO
MFU percentage.

## Next allocation gates

1. Run the explicit Neuron Explorer profile with a fresh compiler cache and
   verify populated HLO `model_flops` before publishing exact MFU.
2. Compare the prepared 9x9 per-rank batch-2,048 continuation with and without
   `TORCH_NEURONX_ENABLE_ASYNC_NRT=1`.
3. Capture CPU launch, host-to-device, HBM, tensor-engine, and all-reduce
   intervals in the same steady-state window.
4. Continue from checkpoint 055 only after the comparison has produced a
   stable step-time and MFU record. The next run must preserve a fresh-seed
   baseline for learning-curve interpretation.
5. Re-run the public KataGo ladder after the next learning segment. Only then
   decide whether the 19x19 continuous run should begin.

The prepared launch and checkpoint-sync commands are documented in the
repository README. Checkpoints remain on head01 and are excluded from GitHub.
