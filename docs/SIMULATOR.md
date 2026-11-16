# Stage 1: ABR Simulator

## Purpose

The simulator models a video player downloading one four-second video chunk at
a time over a recorded bandwidth trace. A policy chooses a bitrate before each
download. The environment then advances network time, updates playback buffer
occupancy, measures any stall, and computes the chunk's Quality of Experience
(QoE) reward.

The environment deliberately knows nothing about fixed-bitrate, buffer-based,
or Q-learning policies. Every future policy will call the same `step()` method,
which makes their evaluation comparable.

## One simulation step

For the next video chunk:

1. The policy supplies one bitrate index from 0 through 5.
2. The simulator finds that chunk's encoded size in bytes.
3. It transfers the bytes across consecutive intervals of the selected network
   trace. Trace bandwidth is expressed in Mbit/s.
4. Only 95% of link capacity is treated as payload, matching MERINA.
5. The simulator adds an 80 ms round-trip delay.
6. Playback consumes the existing buffer during the download. If the download
   takes longer than the available buffer, the difference is rebuffering time.
7. The downloaded chunk adds four seconds to the buffer.
8. If the buffer exceeds 60 seconds, the player waits in 0.5-second increments
   and advances through the bandwidth trace while it waits.
9. The simulator returns the resulting measurements and QoE reward.

At the end of a trace, network time wraps back to the beginning, which matches
the upstream simulator. An episode ends after all 49 video chunks have been
downloaded. The caller must explicitly reset the environment before another
episode.

## Defined Stage 1 configuration

These defaults are inherited from MERINA's linear-QoE experiment rather than
new Q-learning choices:

| Setting | Value |
| --- | --- |
| Bitrate ladder | 300, 750, 1200, 1850, 2850, 4300 Kbit/s |
| Video | Mao metadata from the MERINA repository |
| Chunks per video | 49 |
| Chunk playback duration | 4 seconds |
| Maximum playback buffer | 60 seconds |
| Buffer-drain increment | 0.5 seconds |
| Link payload fraction | 0.95 |
| Link round-trip delay | 0.080 seconds |
| Initial previous bitrate | 750 Kbit/s (bitrate index 1) |
| Rebuffer penalty | 4.3 per second |
| Smoothness penalty | 1.0 per Mbit/s of bitrate change |
| Artificial throughput noise | Disabled, as in the active upstream code |

The configuration lives in `src/abr_simulator/config.py` so every future policy
uses the same values.

## QoE reward

For each chunk, bitrate values are converted from Kbit/s to Mbit/s and the
linear reward is

```text
reward = current_bitrate_Mbps
         - 4.3 * rebuffer_seconds
         - abs(current_bitrate_Mbps - previous_bitrate_Mbps)
```

The first chunk compares its bitrate with MERINA's configured default bitrate
of 750 Kbit/s. Episode QoE is the sum of all chunk rewards.

## Step output

`ABREnvironment.step(action_index)` returns a `StepResult` containing:

- selected bitrate and chunk size;
- total download time, including RTT;
- buffer and rebuffering time after the download;
- time spent waiting because the buffer was full;
- measured download throughput;
- the chunk QoE reward;
- chunks remaining and whether the episode is complete; and
- all available encoded sizes for the next chunk.

Measured throughput is the downloaded bits divided by total download time,
including RTT. It is an observation derived from a completed download; the
future bandwidth values in the trace are never exposed to a policy.

## Imported data

The local data mirrors two folders from the bandwidth-trace repository:

| Local directory | Upstream directory | Files | Current use |
| --- | --- | ---: | --- |
| `data/traces/fcc_hsdpa/train` | `fcc_and_hsdpa/cooked_traces` | 166 | Training candidates |
| `data/traces/fcc_hsdpa/holdout` | `fcc_and_hsdpa/test_traces` | 262 | Untouched split candidates |

Each file contains whitespace-separated `timestamp_seconds bandwidth_Mbps`
rows. A zero bandwidth value represents a real outage and is supported by the
simulator.

The `holdout` name is intentional. Stage 1 does **not** yet decide which of
those files will be validation traces and which will be final test traces.
That deterministic split will be made before tuning Q-learning, so the final
test subset remains unseen.

## Adaptations from the upstream API

The network and buffer mechanics follow MERINA's `envs/env_log.py`. The local
version changes the interface to make experiments safer and easier to test:

- `reset()` and `step()` replace a method that automatically changed to a
  random trace at the end of a video.
- Trace choice and starting position are controlled by the caller.
- Data validation reports malformed traces and video metadata early.
- QoE is calculated in one shared function instead of separately in every
  policy.
- The full buffer sleep duration is preserved in the step result.
- The implementation uses only the Python standard library.

State discretization, baseline thresholds, Q-learning hyperparameters, and the
final validation/test split are intentionally outside Stage 1.

## Verification

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/smoke_simulator.py
```

The tests cover constant bandwidth, a zero-bandwidth outage, rebuffering,
buffer capping and sleep, episode termination, imported file counts, and a
complete 49-chunk run on a real trace.
