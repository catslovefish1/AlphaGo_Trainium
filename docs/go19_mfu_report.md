# 19x19 preparation MFU sweep

Job 16546 used all 64 logical workers and two measured training steps per
candidate. The results are timing proxies with the analytic three-forward
FLOPs estimate. Exact MFU requires Neuron Explorer `model_flops`.

| channels | residual blocks | batch | first step (s) | steady step (s) | aggregate samples/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 3 | 16 | 26.250 | 17.625 | 58.10 |
| 64 | 6 | 16 | 51.500 | 33.000 | 31.03 |
| 64 | 6 | 32 | 51.250 | 32.250 | 63.50 |

Increasing the batch from 16 to 32 nearly doubled aggregate samples/s for the
64-channel, 6-block network without increasing steady step time. The current
19x19 training candidate is therefore 64 channels, 6 blocks, batch 32. The
32-channel, 3-block point remains available as a compile and control variant.

Raw metrics are in `docs/results/16546_go19_mfu/`.
