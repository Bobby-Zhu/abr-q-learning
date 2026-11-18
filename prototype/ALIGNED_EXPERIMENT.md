# Shared simulator experiment

This run trains tabular Q-learning from scratch on the shared simulator and applies the same test traces and metrics to all three policies. The [original experiment](EXPERIMENT.md) remains available with its original data and settings.

## Data and protocol

The split builder audited all 428 imported traces. The previous validation and test manifests shared 36 capture IDs. The revised manifests contain 166 training traces in 166 captures, 130 validation traces in 19 captures, and 132 test traces in 20 captures. Capture IDs and normalized content fingerprints are disjoint across the splits. All input traces passed checks and none were exact duplicates. See the [capture audit](../data/splits/capture_audit.json).

The fixed policy selects 1,200 Kbit/s for every chunk. The buffer policy has a 10-second reservoir and a 40-second cushion. The simulator counts the first bitrate change from 750 Kbit/s for all policies.

The Q-table has 180 states for completed-download observations and a separate startup state. Training uses seed 42, learning rate 0.1, discount 0.95, and exploration probability `max(0.05, 0.5 * 0.92 ** (epoch - 1))`. The state uses the latest completed download's throughput and bitrate, and the current buffer. Q-learning selects the first action from its startup state. Training runs for 40 epochs and selects epoch 6 by mean validation chunk QoE, which was 0.794285 among the six checkpoints. The Q-table is frozen for test evaluation.

## Test results

| Policy | Mean chunk QoE | Capture-weighted mean chunk QoE | Bitrate (Mbps) | Stall (s/video) |
| --- | ---: | ---: | ---: | ---: |
| Fixed | -3.873830 | -4.058772 | 1.200 | 57.713 |
| Buffer | 0.835158 | 0.816583 | 1.135 | 1.542 |
| Q-learning | 0.846741 | 0.821175 | 1.262 | 2.275 |

Q-learning improves mean chunk QoE by 0.011583 over the buffer baseline in this run. Equal weighting of the 20 captures reduces the difference to 0.004592. Q-learning scores higher on FCC traces (1.124774 vs. 0.982580), while the buffer policy scores higher on non-FCC traces (0.761447 vs. 0.707725). Q-learning selects a higher average bitrate overall and incurs more rebuffering. The result covers one video, one training seed, and the selected split.

Of the 20 current test captures, 16 were in the original prototype's training split, one was in its validation split, and two were in its test split. The revised run uses its own capture-disjoint split and a new Q-table. Report this as a comparison under the revised protocol on previously studied data.

## Reproduce

From the repository root with Python 3.10 or later:

```bash
python scripts/create_splits.py --check
python scripts/run_q_learning.py --out prototype/results/run_replay
```

Use a new output directory. The [run archive](results/run_aligned/) includes the Q-table, checkpoint scores, per-trace metrics, summary and config with split hashes. See the original report for [upstream attribution and AI assistance disclosure](EXPERIMENT.md).
