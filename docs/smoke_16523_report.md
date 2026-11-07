# 64-rank smoke report

Run `16523_go9_smoke_v4` completed on `trn09` with 64 XLA workers under LNC2.

## Device gate

- `WORLD_SIZE=64`
- `NEURON_LOGICAL_NC_CONFIG=2`
- `NEURON_RT_NUM_CORES=64`
- all-reduce and model optimizer step passed
- steady-state probe step time was `0.20047473907470703` seconds after compilation
- the fixed-shape analytic timing proxy was `4.42381275979868` aggregate BF16 TFLOP/s against the `10,700` TFLOP/s Trn2.48xlarge BF16 reference

## Learning points

| iteration | policy loss | value loss | random wins as black | random games |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.4375 | 1.09375 | 0 | 512 |
| 2 | 4.40625 | 1.00000 | 0 | 512 |
| 3 | 4.40625 | 1.00000 | 0 | 512 |
| 4 | 4.40625 | 0.9921875 | 0 | 512 |
| 5 | 4.40625 | 0.98828125 | 0 | 512 |
| 6 | 4.40625 | 0.9921875 | 0 | 512 |

Policy and value losses moved downward from initialization. The short random evaluation remained at 0 wins, so this run demonstrates an early optimization signal and synchronized execution rather than playing strength.

The complete retrieved artifacts remain under `runs/16523_go9_smoke_v4/` locally and on head01. The repository includes this report and the text logs, while checkpoint binaries stay in the run directory for local follow-up.

The probe records a whole-node timing proxy. It does not claim an exact MFU because this smoke run did not collect Neuron Explorer HLO `model_flops`. A Neuron profile with HLO `model_flops` is the follow-up measurement for a production-sized model.

KataGo and other checked public Go-engine binaries were not available on head01 or `trn09`, so the optional public-engine match was not run.
