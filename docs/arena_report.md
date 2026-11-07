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

## Initial external result

Fresh-seed checkpoint `checkpoint_030.pt` from job `16535` was matched against
the official KataGo 9x9 network. The 32-visit two-game match scored `0-2`, with
mean game length `105.5` plies. A shorter 8-visit four-game match scored `0-4`,
with mean game length `106.5` plies. Colors alternated in both matches. These
are baseline strength results, not a claim that the short smoke-trained model
is close to KataGo.

Raw results are under `docs/results/16540_katago_v3/` and
`docs/results/16541_katago_fast/`.
