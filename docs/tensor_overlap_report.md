# CPU, HBM, and Neuron forward timing

`alphago.tensor_overlap_probe` measures five fixed-shape paths on all 64
logical NeuronCores. It records host batch preparation, host-to-HBM transfer,
model forward execution, a sequential host→HBM→forward step, and a producer
thread that prepares the next host batch while the device consumes the current
batch.

The reported pipeline speedup is a bounded timing comparison. It shows how
much host preparation can be hidden behind device work. A Neuron device
timeline is still required to attribute time precisely between HBM movement,
tensor-core execution, and lazy XLA synchronization.

Job `16539` ran this probe with 64 ranks, shape `[4, 4, 9, 9]`, two warmups,
and four measured steps. The mean host→HBM time was `0.2965 ms`, the mean
forward time was `3.8071 ms`, and the sequential step was `4.5801 ms`. The
producer-thread pipeline was `3.8130 ms`, giving a `1.2012x` timing proxy and
`0.7671 ms` of hidden headroom. The raw JSON and Slurm log are under
`docs/results/16539_tensor_overlap_v1/`.
