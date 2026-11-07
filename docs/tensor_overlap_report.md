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
