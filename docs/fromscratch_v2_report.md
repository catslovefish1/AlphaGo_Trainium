# Fresh-seed 64-rank learning run

Job `16535` started with random model weights and seed `23`. It did not pass a
checkpoint to the launcher. All 64 logical NeuronCores initialized, trained,
evaluated, and exited cleanly under Slurm state `COMPLETED`.

The short run recorded 34 complete self-play updates plus a final partial
budget-bound update. Policy loss moved from `4.4375` at iteration 2 to
`3.9375` at iteration 35. Value loss moved from `0.69921875` to
`0.2578125`. The final complete update at iteration 34 had policy loss
`3.953125` and value loss `0.255859375`. Random-opponent wins were sparse,
with 3 wins in one 512-game evaluation and zero at most other points.

This is the primary from-scratch smoke result. The loss movement confirms
early target fitting and distributed optimizer progress. It does not establish
playing strength. Readable artifacts are under
`docs/results/16535_fromscratch_v2/`.
