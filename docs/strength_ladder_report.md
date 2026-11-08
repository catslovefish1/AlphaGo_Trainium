# Public-engine strength ladder

The checkpoint was the fresh 9x9 from-scratch checkpoint 030 from job 16535.
Each match used two alternating-color games against the official KataGo 9x9
network and a one-thread search.

| KataGo max visits | our wins | KataGo wins | mean game length |
| ---: | ---: | ---: | ---: |
| 1 | 0/2 | 2/2 | 108.5 |
| 8 | 0/2 | 2/2 | 102.5 |

The checkpoint did not yet beat the very-weak one-visit reference. This is a
baseline for the tuned 9x9 continuation, not a final strength claim. The
earlier 32-visit record was submitted before match command files were isolated
and is retained only as a diagnostic. The launcher now gives every pending
match its own command file so the visit count cannot be overwritten by a later
submission.

Raw match records are in `docs/results/16560_katago_vweak2/` and
`docs/results/16561_katago_weak2/`.
