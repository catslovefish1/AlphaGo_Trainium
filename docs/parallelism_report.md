# Parallelism and fused-NEFF observations

## Native forward compilation

Job `16525` compiled the fixed-shape policy-value forward graph with
`torch_neuronx.trace` on `trn09`.

- batch shape `[64, 4, 9, 9]`
- compile time `9.242753267288208` seconds
- output graph `neuron::forward`
- NEFF artifact stays in the node/head01 run directory and is not committed with source or reports

This gives a concrete native compilation baseline for comparing XLA lazy
execution with a saved native forward graph.

## CPU and collective timing

Job `16528` launched 64 ranks with a warmed-up fixed `[2, 4, 9, 9]` evaluation batch.

- sequential CPU/device loop `0.028120302129536867` seconds
- producer-thread pipeline `0.02717227814719081` seconds
- observed speedup `1.03488938164149x`
- 1024-element float32 all-reduce `0.0002448759041726589` seconds

Job `16527` was the first-use comparison and included compilation or cache
effects in its timing. The warmed-up result is the baseline for future loop
changes. The pipeline result is a timing observation for this shape and
workload. It does not prove device overlap without a timeline trace.

The collective result should be measured alongside model work when deciding
whether to increase data parallelism, batch size, or gradient accumulation.

Job `16529` measured the warmed-up host and collective path over five samples
with 4,096 float32 elements.

- host batch preparation mean `0.08904160931706429 ms`
- host-to-device mean `0.23799999617040157 ms`
- all-reduce mean `4.92686522193253 ms`
- sequential transfer-plus-collective proxy `5.164865218102932 ms`
- pipeline lower-bound headroom `0.23799999617040157 ms`

The measured host preparation is small beside collective latency at this
payload. Larger gradient buckets and accumulation are the next communication
experiments before changing the self-play scheduler.

The first scaled learning trial, job `16530`, used `train_batch=256`. It
compiled its training graph but failed during the first gradient synchronization
with `SyncTensorsGraph` and collective-resource errors. The failure log is
preserved under `docs/results/16530_learning_v1/`. The follow-up keeps 64 ranks
and returns to `train_batch=64`, while increasing search and update counts in
smaller steps.
