# CPU-to-Neuron async pipeline

The queue-depth probe used all 64 logical NeuronCores with per-rank batch 4,096. It compared sequential host preparation plus transfer and forward against a CPU producer feeding queue depths 1, 2, and 4.

| Path | Mean step ms | Speedup |
| --- | ---: | ---: |
| Sequential | 34.294 | 1.000x |
| Queue depth 1 | 30.625 | 1.120x |
| Queue depth 2 | 29.913 | 1.146x |
| Queue depth 4 | 29.721 | 1.154x |

Queue depth 2 was almost as effective as depth 4 and avoids extra host buffering. This probe measures overlap of CPU preparation with the transfer and Neuron consumer path. A device timeline is still required to attribute exact HBM and tensor-core overlap.

## Next training switch

The training launchers now accept an `async_nrt` argument. Passing `1` exports `TORCH_NEURONX_ENABLE_ASYNC_NRT=1`, which enables the Neuron runtime path for overlapping compute and communication. The next 9x9 continuation should compare this setting with the queue-depth result before it is used for 19x19.
