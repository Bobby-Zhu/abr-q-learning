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
status. No baseline or Q-learning policy has been implemented yet.

See [docs/SIMULATOR.md](docs/SIMULATOR.md) for the simulation rules and the
configuration inherited from MERINA.

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
src/abr_simulator/          Simulator, loaders, configuration, and QoE
data/video_sizes/mao/       49 chunk sizes at each of six bitrate levels
data/traces/fcc_hsdpa/      Upstream training and untouched holdout traces
scripts/smoke_simulator.py  Full-video fixed-bitrate smoke run
tests/                      Simulator mechanics and imported-data checks
docs/SIMULATOR.md           Simulator contract and design notes
third_party/licenses/       Required upstream MIT licenses
```

Third-party provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
