# 9x9 architecture MFU sweep

Job 16545 used all 64 logical workers with batch size 64 and two measured
training steps per shape. The probe reports an analytic timing proxy, so these
values are useful for comparing compile and steady-step behavior. Exact MFU
requires the Neuron Explorer profile.

| channels | residual blocks | first step (s) | steady step (s) | aggregate samples/s |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 3 | 26.000 | 14.875 | 275.36 |
| 64 | 6 | 0.504 | 0.224 | 18,315.74 |
| 64 | 12 | 86.500 | 45.000 | 91.02 |
| 128 | 6 | 137.000 | 66.000 | 62.06 |

The 64-channel, 6-block point was warm in the compiler cache when its array
task ran, so its unusually short first step is not a clean compile-time
comparison. The larger variants show the expected higher compile and steady
step cost. The 64-channel, 6-block shape remains the current 9x9 reference
because it already produced the from-scratch learning run and the KataGo
evaluation checkpoint.

Raw metrics are in `docs/results/16545_nn_mfu/`.

The exact MFU profile also includes a system trace for CPU launch and runtime
activity and a device trace for DMA, HBM, synchronization, and tensor-engine
activity. Neuron Explorer's MFU percentage is intentionally device-only. It
does not add CPU-to-device waiting time to the MFU numerator or denominator.
The separate tensor-overlap probe and the system/device trace are the evidence
used to quantify that waiting time and identify whether it overlaps with
device computation.
