# Checkpoint-backed strength path

The first smoke implementation used a first-legal `genmove` and a greedy
policy-only random evaluation. That was enough to validate rules and training
collectives, but it could not measure the learned search policy or provide a
meaningful human-play interface.

`alphago.arena` now loads a rank-0 checkpoint, evaluates fixed-shape policy and
value batches, runs PUCT search without root noise, and records wins, draws,
and mean game length. It supports model-versus-random and checkpoint-versus-
checkpoint matches. `alphago.gtp` uses the same MCTS selector whenever a
checkpoint is supplied.

`alphago.gtp_match` drives our engine and an external GTP engine in alternating
colors. The match harness treats the external engine as the strength reference
and records the exact command, board size, komi, win counts, and game lengths.
The node launcher fixes KataGo to one search thread and 32 visits so the first
comparison has a declared compute budget on both sides.

The arena is intentionally single-core for evaluation. Training remains the
64-rank path. This keeps the strength measurement independent of the training
collective and leaves the remaining logical cores available for a future
training allocation.
