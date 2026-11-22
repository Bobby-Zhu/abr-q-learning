# Trace dynamics: generated results

Common window: 5 seconds. Forecasts use only past windows.
Full definitions, caveats, and interpretation: [TRACE_DYNAMICS.md](../../docs/TRACE_DYNAMICS.md).
This is a descriptive follow-up using frozen splits and saved policy results.
No agent is trained, tuned, or reevaluated by this script.

## Train split

These entries are medians across trace files. NMAE is normalized MAE; lower is better.

| Source | Traces / recordings | Native interval (s) | Lag-1 r | CV | Normalized change | h=1 NMAE | h=3 NMAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCC | 39 / 39 | 5.000 | 0.393 | 0.211 | 0.115 | 0.115 | 0.137 |
| HSDPA | 127 / 127 | 1.011 | 0.782 | 0.421 | 0.206 | 0.205 | 0.244 |

Counts with missing/undefined statistics:

- FCC: 0 undefined correlations; 0 undefined normalized forecast errors.
- HSDPA: 0 undefined correlations; 0 undefined normalized forecast errors.

## Validation split

These entries are medians across trace files. NMAE is normalized MAE; lower is better.

| Source | Traces / recordings | Native interval (s) | Lag-1 r | CV | Normalized change | h=1 NMAE | h=3 NMAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCC | 43 / 6 | 5.000 | 0.221 | 0.188 | 0.113 | 0.113 | 0.156 |
| HSDPA | 86 / 13 | 0.903 | 0.484 | 0.245 | 0.198 | 0.215 | 0.249 |

Counts with missing/undefined statistics:

- FCC: 0 undefined correlations; 0 undefined normalized forecast errors.
- HSDPA: 7 undefined correlations; 7 undefined normalized forecast errors.

## Test split

These entries are medians across trace files. NMAE is normalized MAE; lower is better.

| Source | Traces / recordings | Native interval (s) | Lag-1 r | CV | Normalized change | h=1 NMAE | h=3 NMAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FCC | 45 / 6 | 5.000 | 0.147 | 0.210 | 0.177 | 0.171 | 0.193 |
| HSDPA | 88 / 14 | 1.150 | 0.405 | 0.240 | 0.197 | 0.195 | 0.219 |

Counts with missing/undefined statistics:

- FCC: 0 undefined correlations; 0 undefined normalized forecast errors.
- HSDPA: 0 undefined correlations; 0 undefined normalized forecast errors.

## Files and interpretation

- `per_trace.csv`: all trace statistics, forecast target counts, and saved QoE differences.
- `per_recording.csv`: means of trace metrics within each original recording.
- `summary.csv` / `summary.json`: mean, median, IQR, and valid counts, both by trace and by recording.
- `qoe_associations.csv`: descriptive correlations with saved Q-learning minus buffer QoE.
- `metadata.json`: settings and SHA-256 fingerprints of input data/results.

Adjacent slices from one recording are dependent; no independent-trace significance test is implied.
Lag-one correlation describes persistence, while forecast error measures a particular predictor.
The two can disagree. These trace-level forecasts are proxies for the agent's chunk measurements.
