# Larger-batch HBM sweep

Job 16564 ran one 64-rank point at a time on the 9x9, 64-channel, 6-block
network. Each point included the MFU timing probe, a size-matched all-reduce
probe, and `neuron-monitor` samples. All three points completed both probes
without OOM or collective failure.

| batch | peak Neuron runtime bytes | compile/first step (s) | measured steady step (s) | aggregate samples/s | all-reduce (ms) | host→device (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 189,388,102 | 0.684 | 0.192 | 42,581.77 | 5.221 | 0.297 |
| 256 | 269,538,564 | 61.250 | 20.000 | 819.20 | 2.338 | 0.466 |
| 512 | 508,025,092 | 87.500 | 23.375 | 1,401.84 | 2.317 | 0.904 |

The batch-128 point reused a warm compiler graph. The larger points include
their compilation tail in the first-step field, and their three measured
steps are still too short to use as a final steady-state MFU ranking. Batch
512 stayed stable and increased measured aggregate throughput over batch 256,
while using about 2.68 times the batch-128 runtime memory. A warmed rerun is
needed before selecting the production batch.

The peak-memory field is the maximum `neuron_runtime_used_bytes.neuron_device`
value emitted by `neuron-monitor`. It records observed runtime device bytes
and does not claim a full physical-HBM capacity percentage. Raw monitor logs
remain on trn09 because they are large. Compact MFU and overlap JSON files are
in `docs/results/16564_hbm_batch/`.

The warmed rerun used two warmup steps and five measured steps after the
compiler graphs were already available.

| batch | warmed steady step (s) | warmed aggregate samples/s | all-reduce (ms) | host→device (ms) | peak runtime bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.05005 | 327,360 | 1.943 | 0.600 | 265,384,734 |
| 512 | 0.05298 | 618,515 | 2.569 | 1.124 | 508,025,092 |

The tuned 9x9 continuation uses batch 512 with two optimizer steps per
iteration. The warmed data favors batch 512 for accelerator throughput while
the recorded runtime-memory value remains small relative to the node's HBM
capacity.
