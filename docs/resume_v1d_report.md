# Extended checkpoint continuation diagnostic

Job `16534` resumed checkpoint 20 from the earlier 64-rank run and regenerated
replay. It reached iteration 34 with synchronized 64-rank updates before the
12-minute Slurm allocation timed out during additional XLA compilation.

The continuation moved policy loss from `4.4375` at iteration 2 to `3.84375`
around iteration 30 and value loss from `0.69921875` to `0.1494140625` by
iteration 34. The timeout is retained as a compilation-budget diagnostic.
The resume path is ancillary to the from-scratch result in
[`docs/fromscratch_v2_report.md`](fromscratch_v2_report.md).

Readable artifacts are under `docs/results/16534_resume_v1d/`.
