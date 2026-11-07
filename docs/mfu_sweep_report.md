# Empirical MFU sweep

The first batch-size sweep ran on all 64 logical NeuronCores with the 9x9
`64-channel × 6-block` network. The measured points were

| batch per rank | initial step | measured steady window | aggregate samples/s | analytic MFU proxy |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 46.25 s | 16.50 s | 124.1 | 0.00025% |
| 64 | 0.504 s | 0.202 s | 20,262.3 | 0.0410% |
| 128 | 56.75 s | 18.13 s | 452.0 | 0.00091% |

The batch 64 point reused a compiled graph from earlier jobs. The batch 32 and
128 windows still included compilation work, so those values are diagnostic of
cache and shape behavior rather than final steady-state throughput. Later
probes record every steady-step sample to separate compilation tails from
steady execution.

Raw JSON and logs are under `docs/results/16542_mfu_sweep/`.
