# Final comparison on the validation split

Q-learning cells are mean ± standard deviation over 5 independently trained agents (seeds 100, 101, 102, 103, 104). Baselines are deterministic.

## All traces (129)

| Policy | Mean QoE per video | Avg bitrate (Kbps) | Rebuffer per video (s) | Switches per video |
|---|---|---|---|---|
| Fixed 1200 Kbps | -230.68 | 1200 | 67.22 | 1.00 |
| Buffer-based (10 s / 40 s) | 43.94 | 1217 | 1.75 | 13.09 |
| Q-learning | 47.88 ± 1.87 | 1375 ± 11 | 2.22 ± 0.08 | 12.19 ± 1.68 |

## FCC traces (43)

| Policy | Mean QoE per video | Avg bitrate (Kbps) | Rebuffer per video (s) | Switches per video |
|---|---|---|---|---|
| Fixed 1200 Kbps | -372.63 | 1200 | 100.23 | 1.00 |
| Buffer-based (10 s / 40 s) | 33.38 | 1018 | 1.83 | 14.00 |
| Q-learning | 39.87 ± 1.09 | 1117 ± 14 | 1.86 ± 0.03 | 9.57 ± 1.16 |

## HSDPA traces (86)

| Policy | Mean QoE per video | Avg bitrate (Kbps) | Rebuffer per video (s) | Switches per video |
|---|---|---|---|---|
| Fixed 1200 Kbps | -159.71 | 1200 | 50.71 | 1.00 |
| Buffer-based (10 s / 40 s) | 49.21 | 1316 | 1.71 | 12.64 |
| Q-learning | 51.89 ± 2.36 | 1504 ± 10 | 2.39 ± 0.12 | 13.49 ± 2.07 |
