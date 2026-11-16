# Baseline policies

## Fixed-bitrate baseline

`FixedBitratePolicy` selects the same bitrate for every chunk regardless of
buffer occupancy or network throughput. The declared default is bitrate index
2, or 1200 Kbit/s. It provides a simple reference point for determining
whether adaptation itself improves QoE.

The policy has no training parameters. The command-line script accepts another
bitrate index for exploratory comparisons, but the primary fixed baseline must
remain 1200 Kbit/s when final results are reported.

Run it over all training traces with:

```bash
python3 scripts/run_fixed_baseline.py
```

The implementation is in `src/abr_simulator/policies/fixed_bitrate.py`.

## Buffer-based baseline

`BufferBasedPolicy` is a deterministic reference policy. It does not train,
estimate bandwidth, inspect future trace values, or use encoded chunk sizes.
Its only input is the playback buffer immediately before choosing the next
chunk's bitrate.

The policy uses two fixed regions:

- **Reservoir:** 10 seconds. At or below this level, select 300 Kbit/s.
- **Cushion:** 40 seconds. Between 10 and 50 seconds, map buffer occupancy
  linearly onto the six-level bitrate ladder.
- **Upper threshold:** 50 seconds. At or above this level, select 4300 Kbit/s.

With the current bitrate ladder, the action boundaries are:

| Buffer before download | Selected bitrate |
| ---: | ---: |
| Less than 18 seconds | 300 Kbit/s |
| 18 to less than 26 seconds | 750 Kbit/s |
| 26 to less than 34 seconds | 1200 Kbit/s |
| 34 to less than 42 seconds | 1850 Kbit/s |
| 42 to less than 50 seconds | 2850 Kbit/s |
| 50 seconds or more | 4300 Kbit/s |

The exact action rule for a buffer strictly between the reservoir and upper
threshold is

```text
fraction = (buffer_seconds - 10) / 40
action = floor(fraction * 5)
```

The first action is 300 Kbit/s because an episode starts with an empty buffer.
The shared QoE calculation still compares that action with MERINA's initial
previous bitrate of 750 Kbit/s, so the initial change is counted consistently
for every policy.

The implementation is in
`src/abr_simulator/policies/buffer_based.py`. The values are command-line
configurable for exploratory validation, but 10 and 40 seconds are the
declared default baseline and must not be tuned using the final test subset.

## Evaluation metrics

Every completed trace produces:

- total and mean per-chunk QoE;
- average selected bitrate in Kbit/s;
- total rebuffering time in seconds;
- number of bitrate changes; and
- total and average magnitude of bitrate changes in Kbit/s.

Switching statistics include the transition from the configured initial
previous bitrate to the first selected bitrate. This matches the smoothness
term already included in the QoE reward.

Aggregate output reports mean episode QoE, mean chunk QoE, average bitrate,
mean rebuffering per episode, mean switches per episode, and average bitrate
change per chunk. All current videos contain 49 chunks, but aggregation still
weights chunk-level quantities by their actual chunk counts.

## Buffer-baseline commands

Evaluate all 166 training traces with the declared baseline:

```bash
python3 scripts/run_buffer_baseline.py
```

Print per-trace results or run a short smoke evaluation:

```bash
python3 scripts/run_buffer_baseline.py --limit 3 --per-trace
```

Both baseline scripts support the declared `train`, `validation`, and `test`
manifests. Validation may be used for development choices. Do not run or
inspect test results until every policy and hyperparameter is frozen.
