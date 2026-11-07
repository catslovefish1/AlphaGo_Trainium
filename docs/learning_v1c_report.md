# 64-rank learning follow-up

Job `16532` ran `configs/go9_learning_v1b.json` on `trn09` with 64 XLA
workers. The first iteration waited for every rank to reach the replay gate.
Iterations 2 through 20 completed with synchronized updates.

- 19 post-start metric points
- 64-rank training and evaluation collectives completed
- policy loss moved from `4.4375` to `4.0625`
- value loss moved from `0.486328125` to `0.216796875`
- random evaluation used 512 games per point
- rank-0 checkpoints were saved at iterations 5, 10, 15, and 20

Random-opponent wins were sparse and did not provide a stable strength signal
in this short run. The loss movement and clean collective execution establish
the next baseline for longer resumed training.

The readable artifacts are under `docs/results/16532_go9_learning_v1c/`. The
rank-0 checkpoints remain on head01 under
`/home/ralph/zhiwei_wang/AlphaGo_Trainium/runs/16532/go9_learning_v1c/`.
