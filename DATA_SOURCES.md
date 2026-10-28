# Data sources and attribution

## Upstream human experiment

The human experiment is McKee et al., *Scaffolding cooperation in human groups with deep reinforcement learning*, Nature Human Behaviour (2023), [doi:10.1038/s41562-023-01686-7](https://doi.org/10.1038/s41562-023-01686-7). The original datasets and analysis scripts are hosted at [OSF 8ahkg](https://osf.io/8ahkg/).

Upstream copyright: DeepMind Technologies Limited, 2022. The [upstream README](https://osf.io/download/5k6es/) specifies [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) for its software and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for other materials. Those terms remain attached to upstream materials. The human-data calibration summaries in this package are derived reanalyses; they are not new human observations or outputs endorsed by the original authors. Published behavioural parameters and distilled recommendation tables used by the simulator are attributed to that work.

## Inputs for calibration

Download the following nine upstream files, preserving their filenames, into `osf/`. Raw human input files and participant-level reconstructed records are not bundled in this release. These download links and SHA-256 values were read from the public OSF API and verified against the downloaded bytes on 28 October 2026.

| Filename | Bytes | SHA-256 |
| --- | ---: | --- |
| [baseline_cooperation_data.csv](https://osf.io/download/av4n8/) | 395201 | `0f37df0b399b68c26040a87934af358e8b6baccb46f6178addbc2051c196df85` |
| [baseline_graph_struct_data.csv](https://osf.io/download/wpj9v/) | 2380989 | `96202bd110e6be780642b5a33d9ebf77ec091dd49a739575baf6d8c9640f4ab0` |
| [baseline_group_outcomes_data.csv](https://osf.io/download/grkw9/) | 99687 | `8eba856d55cbc3a79a036ecd8d6a61768248c76229df4d456c952c4d904076c1` |
| [evaluation_cooperation_data.csv](https://osf.io/download/sehnz/) | 114628 | `25cb1d934b60e1f03c476fff674a3066d0d89fbafdcb18e701585a6bda5ca6b8` |
| [evaluation_graph_struct_data.csv](https://osf.io/download/mcfg9/) | 785542 | `fd25e5f1b63714424d988d34748addf980ad971fc6f9229aec463ab95248fa07` |
| [evaluation_group_outcomes_data.csv](https://osf.io/download/vp3f9/) | 36964 | `600f8ebe886178ce8f5dd9015f80abe0469cec3406b9a5a62e0751f9a69e4fc6` |
| [validation_cooperation_data.csv](https://osf.io/download/gy3qa/) | 349171 | `33cc6e254882246286effb7c36603f695c1cf741e5fd28992f1ff725d1a19e97` |
| [validation_graph_struct_data.csv](https://osf.io/download/wxn35/) | 2356511 | `6add53314d27325f8f7e8a2acd53905391f413848412db63c25b6f4200386d51` |
| [validation_group_outcomes_data.csv](https://osf.io/download/wthq7/) | 107816 | `f805fc1900a9b4fa112743c366b7f3d6cdf846271ced30baa3b8744711184e7f` |

## Included derived data

The files below are copied byte-for-byte from the supplied revised manuscript archive. Their release hashes are in `SHA256SUMS`.

| File | Contents and provenance |
| --- | --- |
| `calib_v2.json` | Aggregate fitted coefficients, covariance matrices, cross-validation and bootstrap summaries, sample counts and human group summaries from `calib2.py` |
| `bootstrap_draws.json` | 200 cooperation-model parameter draws for each of count and fraction families |
| `accept_boot_draws.json` | 200 degree-model and within-person acceptance parameter draws |
| `reconstruction_audit.csv` | 87 group-level reconstruction audit rows; no participant decision records |
| `eval_games_v2.csv.gz` | 847,600 synthetic evaluation-game records from `runs2.py` |
| `levdata_n16.csv.gz` | 39,103 synthetic learned-value training examples and paired-rollout labels |
| `levfit_v2.json` | Archived held-out learned-value diagnostics and permutation importance; not a serialized trained model |
| `submod_v2.csv` | Exact one-step selection diagnostics from simulated states |
| `interactions_v2.csv` | Paired simulation-rollout interaction estimates |

All game-level outputs and training examples above are simulated. The calibration outputs and reconstruction audit summarize the upstream human experiment. New fits and new simulations should be written to `work/`, with their environment and provenance retained separately.

## Integrity check

From the repository root, run:

```text
python -c "from pathlib import Path; import hashlib; pairs=[line.split('  ',1) for line in Path('SHA256SUMS').read_text().splitlines()]; bad=[name for digest,name in pairs if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]; assert not bad,bad; print('All release files match SHA256SUMS')"
```

The manifest covers every release file except itself. Git attributes preserve file bytes across checkout so that line-ending conversion does not invalidate the checksums.
