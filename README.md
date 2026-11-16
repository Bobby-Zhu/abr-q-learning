# ABR Q-Learning

This project studies whether a tabular Q-learning policy can select video
bitrates under changing network conditions more effectively than simple
fixed-bitrate and buffer-based policies.

## Current status

Stage 1 provides a tested, policy-independent ABR simulator adapted from the
MIT-licensed [MERINA](https://github.com/confiwent/merina) virtual player.
It includes the proposal-relevant FCC/HSDPA traces and Mao video chunk-size
metadata from the authors' repositories.

The simulator accepts a bitrate index for each chunk and returns download
delay, buffer level, rebuffering, measured throughput, QoE reward, and episode
status. A linear buffer-based baseline is implemented; the fixed-bitrate and
Q-learning policies are still future stages.

See [docs/SIMULATOR.md](docs/SIMULATOR.md) for the simulation rules and the
configuration inherited from MERINA.

## Run the buffer-based baseline

The default command evaluates the fixed 10-second reservoir and 40-second
cushion policy on all training traces:

```bash
python3 scripts/run_buffer_baseline.py
```

For a quick single-trace check:

```bash
python3 scripts/run_buffer_baseline.py --limit 1 --per-trace
```

See [docs/BASELINES.md](docs/BASELINES.md) for the policy definition and metric
semantics.

## Run the Stage 1 smoke test

No third-party Python packages are required.

```bash
python3 scripts/smoke_simulator.py
```

To select another fixed bitrate or trace:

```bash
python3 scripts/smoke_simulator.py --bitrate-index 0 --trace data/traces/fcc_hsdpa/train/bus.ljansbakken-oslo-report.2010-09-28_1407CEST.log
```

## Run the tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Repository layout

```text
src/abr_simulator/             Simulator, policies, evaluation, and QoE
data/video_sizes/mao/          49 chunk sizes at each of six bitrate levels
data/traces/fcc_hsdpa/         Upstream training and untouched holdout traces
scripts/smoke_simulator.py     Full-video fixed-bitrate smoke run
scripts/run_buffer_baseline.py  Buffer-baseline evaluation
tests/                         Simulator, policy, and metrics checks
docs/SIMULATOR.md              Simulator contract and design notes
docs/BASELINES.md              Baseline rules and evaluation metrics
third_party/licenses/          Required upstream MIT licenses
```

Third-party provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
