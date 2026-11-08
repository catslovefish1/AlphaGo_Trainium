# 9x9 tuned continuation report

The tuned continuation used all 64 logical NeuronCores on trn09 with the XLA backend, a 64-channel six-block network, per-rank batch 512, two optimizer updates per iteration, and asynchronous optimizer barriers.

## Learning signal

The run resumed from checkpoint 030 and reached checkpoint 040 before the 30-minute walltime. Across iterations 31 to 42, policy loss moved from 3.96875 to 3.875 and value loss moved from 0.1465 to 0.1416. The replay buffer grew from 8,128 to 115,200 positions. The captured point at iteration 42 used world size 64.

A second continuation from checkpoint 040 ran as Slurm job 16573. It reached iteration 45 at world size 64. Policy loss moved from 3.75 at iteration 41 to 3.609375 at iteration 45. Value loss was 0.6758 at iteration 45 after the compile-boundary spike. The job ended cleanly after 36 minutes with no checkpoint 050 because four graph transitions consumed the short budget. Checkpoint 040 remains the latest usable synchronized weight file.

## Public-engine strength check

Checkpoint 040 was tested against the official KataGo 9x9 network with two games per setting.

| KataGo visits | Our wins | KataGo wins | Mean game length |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 2 | 105.5 |
| 8 | 0 | 2 | 109.5 |
| 32 | 0 | 2 | 113.5 |

These are early relative-strength measurements. They show that the smoke-test model is learning from self-play while remaining far below the public engine.

## Performance and MFU context

The warmed HBM sweep favored per-rank batch 512. It measured 508,025,092 observed runtime device bytes, 0.05298 seconds per steady synthetic step, 618,515 aggregate samples per second, 2.5688 ms all-reduce time, and 1.1235 ms host-to-device transfer time. The device-byte field is a runtime observation and is not a percentage of physical HBM.

The Neuron Explorer profiles still report `model_flops: 0`. The current instruction-MFU and HBM activity numbers are retained as diagnostics, but this run does not support an exact HLO MFU claim. Exact MFU remains pending a profile with populated HLO `model_flops`.

## Artifacts

Weights remain on head01 under the synchronized checkpoint directory. Readable metrics and match summaries are stored here so the run can be compared with later checkpoints without copying model files into GitHub.
