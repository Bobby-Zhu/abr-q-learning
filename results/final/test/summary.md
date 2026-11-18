# Final comparison on the test split

Q-learning cells are mean ± standard deviation over 5 independently trained agents (seeds 100, 101, 102, 103, 104). Baselines are deterministic.

## All traces (133)

| Policy | Mean QoE per video | Avg bitrate (Kbps) | Rebuffer per video (s) | Switches per video |
|---|---|---|---|---|
| Fixed 1200 Kbps | -261.37 | 1200 | 74.35 | 1.00 |
| Buffer-based (10 s / 40 s) | 36.78 | 1069 | 1.66 | 14.48 |
| Q-learning | 39.64 ± 1.10 | 1176 ± 7 | 1.97 ± 0.11 | 12.34 ± 1.95 |

## FCC traces (45)

| Policy | Mean QoE per video | Avg bitrate (Kbps) | Rebuffer per video (s) | Switches per video |
|---|---|---|---|---|
| Fixed 1200 Kbps | -269.22 | 1200 | 76.18 | 1.00 |
| Buffer-based (10 s / 40 s) | 38.29 | 1146 | 1.81 | 17.38 |
| Q-learning | 44.44 ± 1.03 | 1269 ± 12 | 1.94 ± 0.12 | 12.29 ± 1.77 |

## HSDPA traces (88)

| Policy | Mean QoE per video | Avg bitrate (Kbps) | Rebuffer per video (s) | Switches per video |
|---|---|---|---|---|
| Fixed 1200 Kbps | -257.35 | 1200 | 73.42 | 1.00 |
| Buffer-based (10 s / 40 s) | 36.01 | 1029 | 1.59 | 13.00 |
| Q-learning | 37.19 ± 1.15 | 1129 ± 5 | 1.99 ± 0.14 | 12.36 ± 2.05 |
