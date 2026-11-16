# Third-party notices

## MERINA

- Source: <https://github.com/confiwent/merina>
- Imported commit: `dfd01b7f664e3a9fa0d123945081f7af2ebb31f7`
- Copyright: 2022 Nuowen Kan
- License: MIT; see `third_party/licenses/MERINA_LICENSE`

The buffer and network-transfer mechanics in `src/abr_simulator/environment.py`
are adapted from `envs/env_log.py`. The files in `data/video_sizes/mao` are
copied from `envs/video_size/Mao` at the commit above.

## Real-world bandwidth traces

- Source: <https://github.com/confiwent/Real-world-bandwidth-traces>
- Imported commit: `d28c4fd48d0e358db024cc0217b111c070815fa7`
- Copyright: 2023 Nuowen Kan
- License: MIT; see
  `third_party/licenses/REAL_WORLD_BANDWIDTH_TRACES_LICENSE`

The local `train` directory is copied from
`fcc_and_hsdpa/cooked_traces`. The local `holdout` directory is copied from
`fcc_and_hsdpa/test_traces`. Unrelated trace collections and the repository's
large backup directory were intentionally not imported.
