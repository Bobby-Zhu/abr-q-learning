# Q-learning prototype

This directory records an exploratory run of tabular Q-learning with MERINA's
deterministic player. It includes the code, selected Q-table, trace manifests,
and paired test metrics from the original run. The shared simulator and baseline
scripts live elsewhere in this repository.

The prototype grouped traces by original recording, selected a fixed bitrate
and training checkpoint on validation, and froze both choices before test
evaluation. The archived experiment uses a validation-selected fixed rate of
0.30 Mbps and a buffer policy with thresholds of 4 and 16 seconds. The shared
baselines use 1.20 Mbps and thresholds of 10 and 50 seconds. Compare the prototype
policies using the archived evaluation settings.

See the [experiment report](EXPERIMENT.md) for the state, reward, results,
limitations, upstream attribution, and AI assistance disclosure.

## Reproduce the archived run

Use Python 3.10 or newer, NumPy, Pillow, and Git. From the repository root:

```bash
python -m pip install numpy "Pillow>=10.1"
git clone --filter=blob:none --no-checkout https://github.com/confiwent/merina.git prototype/upstream/merina
git -C prototype/upstream/merina sparse-checkout set envs
git -C prototype/upstream/merina checkout --detach dfd01b7f664e3a9fa0d123945081f7af2ebb31f7
git clone --filter=blob:none --no-checkout https://github.com/confiwent/Real-world-bandwidth-traces.git prototype/upstream/traces
git -C prototype/upstream/traces sparse-checkout set fcc_ori traces_3gp
git -C prototype/upstream/traces checkout --detach d28c4fd48d0e358db024cc0217b111c070815fa7
python prototype/src/experiment.py --out prototype/results/run_replay
```

Choose a new output directory for each run. A different folder containing the
two pinned checkouts can be supplied with `--upstream`. The test split has
already been evaluated for the archived run.
