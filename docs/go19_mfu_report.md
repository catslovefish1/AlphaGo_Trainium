# 19x19 preparation sweep

Before starting 19x19 self-play, the node will measure three fixed-shape
policy-value training steps on all 64 logical NeuronCores. The candidates are
`32 channels × 3 blocks × batch 16`, `64 × 6 × 16`, and `64 × 6 × 32`.

Each task records first-step compile time, steady-state step time, aggregate
samples/s, and the analytic MFU proxy. The result will select the first 19x19
shape for a bounded self-play smoke run.
