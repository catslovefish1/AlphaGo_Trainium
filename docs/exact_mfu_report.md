# Exact MFU profile

The selected 9x9 shape will run a single-rank training workload under Neuron
Explorer with HLO debug metadata and device profiling enabled. The profile
summary is expected to expose `model_flops`, `mfu_hlo_estimated_percent`, HBM
bytes, tensor-engine time, and the memory-bound upper limit. The probe is
single-rank because Neuron Explorer does not combine multi-rank device profiles
into one unified view. The 64-rank empirical sweeps remain the throughput and
communication measurements.

The system profile is enabled alongside the device profile so runtime calls and
host-side launch or transfer intervals can be compared with the device DMA and
HBM intervals.
