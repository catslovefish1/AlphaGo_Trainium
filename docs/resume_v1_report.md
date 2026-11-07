# Checkpoint resume validation

Job `16533` resumed the rank-0 checkpoint from job `16532` on all 64 logical
NeuronCores. The checkpoint loaded cleanly, the ranks passed the post-load
rendezvous, and the job exited with Slurm state `COMPLETED`.

The run regenerated replay because the source checkpoint was saved with
`checkpoint_replay: false`. The first attempted continuation iteration waited
until every rank had enough local replay. The next synchronized update recorded
iteration 22 with policy loss `3.921875`, value loss `0.1328125`, and 512
random-opponent games. This is a resume and checkpoint-integrity result. It is
not a strength comparison because the replay distribution was regenerated.

Readable artifacts are under `docs/results/16533_resume_v1/`. The source
checkpoint remains on head01 at
`/home/ralph/zhiwei_wang/AlphaGo_Trainium/runs/16532/go9_learning_v1c/checkpoint_020.pt`.
