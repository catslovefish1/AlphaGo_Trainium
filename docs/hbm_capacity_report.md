# 9x9 per-rank batch capacity

The 64-rank sweep used the 64-channel, six-block 9x9 network on trn09 with `NEURON_LOGICAL_NC_CONFIG=2` and `NEURON_RT_NUM_CORES=64`.

| Per-rank batch | Aggregate batch | Stable | Observed runtime device bytes | Warmed samples/s | Host-to-device ms | All-reduce ms |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 512 | 32,768 | yes | 386,723,376 | 2,131,149 | 0.718 | 2.747 |
| 1,024 | 65,536 | yes | 903,833,732 | 3,316,137 | 1.192 | 2.972 |
| 2,048 | 131,072 | yes | 1,712,677,124 | 3,646,351 | 2.939 | 5.483 |
| 4,096 | 262,144 | yes | 3,475,645,188 | 3,539,827 | 6.423 | 10.822 |
| 8,192 | 524,288 | yes | 6,031,384,286 | 1,931,103 | 10.105 | 16.725 |
| 16,384 | 1,048,576 | compile timeout | unknown | unknown | unknown | unknown |

The warmed throughput calculation removes the first steady sample because that sample includes a new graph compile. Batch 2,048 was the best measured point. Batch 4,096 was close and remained stable. The 16,384 run timed out during compilation, with no HBM out-of-memory event observed.

The runtime device-byte field is an observed allocation signal, not a percentage of physical HBM. Exact MFU still requires a Neuron Explorer profile with populated HLO `model_flops`.
