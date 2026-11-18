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
status. Fixed-bitrate and linear buffer-based baselines are implemented along
with deterministic train/validation/test manifests. A tabular Q-learning
agent is trained on the training split, tuned on validation, and compared
with both baselines once on the frozen test split; see
[docs/Q_LEARNING.md](docs/Q_LEARNING.md).

See [docs/SIMULATOR.md](docs/SIMULATOR.md) for the simulation rules and the
configuration inherited from MERINA.

## Run the fixed-bitrate baseline

The declared fixed baseline selects 1200 Kbit/s for every chunk:

```bash
python3 scripts/run_fixed_baseline.py
```

Other ladder entries can be explored explicitly, for example:

```bash
python3 scripts/run_fixed_baseline.py --bitrate-index 0 --limit 1
```

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

## Train, tune, and evaluate Q-learning

No third-party packages are needed for any step.

```bash
# Train one agent on the training split and score it on validation
python3 scripts/train_q_learning.py --episodes 30000 --seed 0

# Hyperparameter search on validation only (round 2, about 20 minutes on 4 cores)
python3 scripts/tune_q_learning.py

# Frozen final comparison on test, then the report figures
python3 scripts/evaluate_final.py --split test
python3 scripts/make_figures.py --split test
```

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
data/traces/fcc_hsdpa/         Imported upstream trace files
data/splits/                   Deterministic train/validation/test manifests
scripts/smoke_simulator.py     Full-video fixed-bitrate smoke run
scripts/run_fixed_baseline.py   Fixed-baseline evaluation
scripts/run_buffer_baseline.py  Buffer-baseline evaluation
scripts/train_q_learning.py    Train one Q-learning agent
scripts/tune_q_learning.py     Validation-only hyperparameter search
scripts/evaluate_final.py      Frozen test comparison of all policies
scripts/make_figures.py        SVG report figures from the final results
results/                       Tuning table, chosen settings, final results
tests/                         Simulator, policy, and metrics checks
docs/SIMULATOR.md              Simulator contract and design notes
docs/BASELINES.md              Baseline rules and evaluation metrics
docs/Q_LEARNING.md             Q-learning design, tuning, and results
third_party/licenses/          Required upstream MIT licenses
```

Third-party provenance is recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
