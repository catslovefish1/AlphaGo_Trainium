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

The first two captures reused the default `/var/tmp/neuron-compile-cache`, so
their profiles retained no HLO FLOP metadata and reported `model_flops: 0`.
The launcher now gives each capture a fresh `NEURON_CC_FLAGS=--cache_dir=...`
directory and enables `NEURON_FRAMEWORK_DEBUG=1` before compilation. A fresh
capture is required before treating `mfu_hlo_estimated_percent` as exact.

The next capture uses a small explicit matmul workload to validate HLO FLOP
accounting. The policy-value network profile remains useful for device,
HBM, DMA, and synchronization timings, but its convolution kernels currently
produce instruction-level MFU without HLO `model_flops` in this SDK build.

The explicit matmul capture also produced `model_flops: 0` while reporting
nonzero MATMUL instruction counts. The installed compiler is NeuronX 2.26.6360
and the Explorer executable is release 2.31. The artifact therefore proves
that the device profile and HLO graph were captured, but the SDK combination
does not expose the requested HLO aggregate. The recorded instruction MFU,
HFU, HBM, DMA, and tensor-engine fields remain valid. A later SDK-matched
profile is required before claiming an HLO-based exact MFU percentage.
