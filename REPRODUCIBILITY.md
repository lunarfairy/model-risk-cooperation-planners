# Reproducibility guide

Run commands from the repository root, using the environment described in the README. Use a fresh output directory for each run; the scripts write fixed filenames and will replace existing outputs there. Keep `data/` as the archived reference results.

## Three distinct workflows

### A. Replot archived results

```text
python figures.py data work/figures
```

All required figure inputs are already included. Fonts can differ by operating system, so PDF/PNG bytes need not match the original rendering even when the numerical tables agree.

### B. Refit the learned value model from its archived examples

```text
python -c "from pathlib import Path; import shutil; Path('work/lev').mkdir(parents=True, exist_ok=True); shutil.copy2('data/levdata_n16.csv.gz', 'work/lev/levdata_n16.csv.gz')"
python runs2.py data work/lev levfit full 1
```

This creates `work/lev/lev16.pkl` and `work/lev/levfit_v2.json`. The original model file is missing from the supplied archive. A newly trained model is a new artifact and cannot be assumed to reproduce the archived LEV games exactly. Use only model files generated locally or from a trusted source; model persistence uses Python pickle.

### C. Full re-estimation and new simulation run

Download the nine files named in [DATA_SOURCES.md](DATA_SOURCES.md) into `osf/`. Create output folders before calibration:

```text
python -c "from pathlib import Path; Path('work/cal').mkdir(parents=True, exist_ok=True); Path('work/full').mkdir(parents=True, exist_ok=True)"
python calib2.py osf work/cal 4 200
python runs2.py work/cal work/full all full 4
python -c "import shutil; [shutil.copy2('work/cal/'+name, 'work/full/'+name) for name in ('calib_v2.json', 'bootstrap_draws.json', 'accept_boot_draws.json')]"
python figures.py work/full work/figures_from_rerun
```

The final calibration copies are necessary: calibration and simulation originally use separate output directories, whereas `figures.py` expects its inputs together. These are example commands for a new, computationally expensive run, not a claim that the full workflow was repeated in this release audit. Adjust the worker count to the available CPU and RAM. Windows `spawn` workers are supported by the calibration patch; full calibration and full simulations were not rerun on Windows or Linux during this audit.

To simulate with archived calibration instead, replace `work/cal` by `data` in the `runs2.py` command and copy the three calibration JSON files from `data` to `work/full` before plotting. Do not mix partial new outputs with the archived evaluation file.

`runs2.py` accepts phases `levdata`, `levfit`, `submod`, `eval`, and `all`. `all` creates training examples, fits `lev16.pkl`, evaluates selection diagnostics, then runs the grid. Running `eval` alone needs `lev16.pkl` already in its output directory. The fourth argument `smoke` reduces the workload; `full` requests the full grid. The fifth argument is the worker count. Reduced-grid outputs are diagnostics and are not sufficient inputs for all manuscript figures. The standalone `smoke_test.py` is the quickest check.

## Model and random-number conventions

The implemented donation game uses benefit 0.10, cost 0.05, initial capital 1 and 15 rounds by default. The default initial graph has expected mean degree 4.5. `runs2.py:grid()` specifies the alternatives, including fixed density 0.3, Chung–Lu and block graphs, and 10/25-round games. The count and fraction models and acceptance coefficients are loaded from `data/calib_v2.json`; published coefficients and recommendation tables are in `engine2.py`.

Calibration uses 40-point Gauss–Hermite integration; planner beliefs use a 25-point disposition grid. Group cross-validation uses seed 1; cooperation bootstraps use seeds 1000+i and acceptance bootstraps use 5000+i. `seed_of()` maps the experiment/world key, network size, game index and experiment family to a deterministic integer using CRC32. Each game then keys random streams by seed, round, purpose and rollout replicate. Learned-value generation uses separate seeds beginning with family 7, and selection diagnostics use family 8. The learned-value train/test split and gradient-boosting estimator both use random state 0. Keep the archived training row order when refitting.

Degree caps govern admission of new edges. They do not delete existing over-cap edges or use simultaneous cuts to create addition capacity. Budgets are K recommendations per round or K=max(1, round(rho*n)). Private reports apply randomized response with truth probability exp(epsilon)/(1+exp(epsilon)); the privacy scope is the reported actions.

## Output fields and figure mapping

`eval_games_v2.csv.gz` contains one game per row. Configuration fields include `exp`, `world`, `n`, `kind`, budget/cap/model information and `game`; outcomes include `final`, `mean`, `capital`, degree statistics and recommendation statistics. `game` is an index within a cell, not a globally unique ID. Reference arms are shared between experiments; the supplied figure script's configuration-key count is 828,998.

**`mean_deg` in the evaluation CSV is the final mean degree.** The summary overwrites the similarly named initial-degree argument when rows are assembled. The initial setting is given by `grid()` and, for the context sweep, `ctx`: `er3` and `er8` use 3 and 8; the other contexts use the default 4.5. Use `p=0.3` for the E5 fixed-density setting. The archived output schema is preserved.

| Output | Main inputs / experiment families |
| --- | --- |
| `fig1_calibration` | Calibration JSON, bootstrap draws, E0 and E0loco |
| `fig2_budget` | E1 budget curves and E8 uncertainty propagation |
| `fig3_information` | E3 information/privacy factorial |
| `fig4_scale` | E4 scaling and E9 mechanism decomposition |
| `fig5_caps_density` | E6 cap sweep and E7 density-response sweep |
| `figS1_selection` | `submod_v2.csv`, `interactions_v2.csv` |
| `figS2_sensitivity` | E10 context and E11 acceptance sensitivity |
| `figS3_robustness_lev` | E2 robustness, E5 fixed density, archived `levfit_v2.json` |
| `tables/*.csv`, `tables/key_numbers.json` | Numerical summaries of the same inputs |

`figures.py` produces 16 CSV tables plus a key-number JSON. It does not typeset manuscript tables. Some missing JSON summary values are serialized as Python-style `NaN`; Python's JSON reader accepts them, but strict JSON consumers may need to convert them to null.

## Release audit: 28 October 2026

| Check | Observed result |
| --- | --- |
| Isolated environment | Python 3.12.4; full package versions in `requirements-lock.txt`; one BLAS/OpenMP thread |
| Upstream input integrity | Nine OSF datasets downloaded; all SHA-256 values matched upstream metadata |
| Reconstruction | Decision, recommendation and group-audit tables matched the supplied archive |
| Sample counts | 87 groups, 1,384 participants, 19,706 decisions; 204 exclusions; 19,502 fitted decisions; 8,251 acceptance records; 1,174 recovered actions; zero undecodable |
| Stored-parameter likelihoods | All six principal fitted specifications agreed within 1e-8 absolute tolerance; no full re-optimization |
| Archived experiment grid | All 7,180 cell counts matched the 847,600 saved records; initial degree identified through context/grid as explained above |
| Original smoke test | Passed: 18 planner configurations, 90 games, plus posterior/value/locality checks |
| Spawn regression | Passed on synthetic records and on one bootstrap draw from all 8,251 acceptance records; serial/worker agreement for both acceptance models |
| Multiprocessing driver | Passed reduced `levdata smoke 2` phase on Windows |
| Saved-output figures | All eight PDF/PNG figures generated; contact-sheet inspection found no missing panels or clipping |
| Numerical summaries | All 16 CSV tables matched at rtol=1e-10, atol=1e-12; key-number JSON numerical values matched exactly |
| Representative reruns | 100 games in 50 cells, covering all experiment families and planner kinds, information protocols, and selected n=512 cells; 98 non-LEV games matched at rtol=1e-9, atol=1e-10 |
| Learned-value refit | 39,103 examples; 7,821 held out. R² 0.0198616 vs archived 0.0199551; held-out correlation 0.141410 vs archived 0.145047; two checked LEV games differed |

The source of the LEV difference is unresolved. The supplied materials contain neither the original trained model nor a frozen training environment, so this audit does not assign a definitive cause. Repeating the fit in the audited environment produced byte-identical model and diagnostic files; the difference from the archive is not established to be ordinary run-to-run randomness. Saved LEV diagnostics and figure inputs remain unchanged. The full experiment grid, complete likelihood optimization, 200-draw bootstraps, and all diagnostic rollout labels were not regenerated.

### Additional check of consequences for the manuscript

All 33 LEV configurations were rerun with the refitted model and the original evaluation seeds: 21 budget cells with 600 games each and 12 robustness cells with 400 games each, totaling 17,400 games. Across cells, the equally weighted mean absolute change in final cooperation was 0.556 percentage points. The largest change occurred in the synthetic unanimity world at K=4: the mean changed from 0.18328125 to 0.13671875, a decrease of 4.656 percentage points (paired SE 0.625 percentage points). That cell difference exceeds ordinary paired sampling uncertainty in this comparison; numerical equivalence is not claimed.

There were two changes of mean ordering among 179 matched comparisons with the archived comparator planners. Both concerned near ties: count world K=2 against unrationed encouragement (LEV gap +0.005208 to -0.000625), and weaker-neighbour-response world K=16 against the published-model planner (gap -0.001406 to +0.000938). In both cases the old and new gaps had descriptive 95% ranges crossing zero. These are descriptive normal approximations, not multiplicity-adjusted tests or equivalence tests.

The manuscript explicitly excludes LEV from its main planner comparisons, and the figure code selects non-LEV planners for those comparisons. Its reported LEV variance explained (2.0%), myopic baseline (1.6%), split-half reliability (0.04), approximate reliability ceiling (7%), and dominant myopic-value feature are preserved at the reported precision. The audit therefore supports retaining those manuscript conclusions while recording the unresolved exact-reproduction difference. Random seeds, source algorithms, and archived results are unchanged; the additional runs are audit outputs rather than replacement paper data.

## Changes to the supplied source

`engine2.py`, `runs2.py`, `figures.py`, `smoke_test.py` and `requirements.txt` are unchanged. In `calib2.py`, the worker initializer explicitly receives recommendation records for the acceptance bootstrap. Previously these records were inherited implicitly through `fork`, leaving them unset in spawned workers. The update changes process initialization only; the equations, objective functions, optimizer settings, random seeds and archived data are preserved. `test_spawn_bootstrap.py` tests the correction.

The release additionally supplies documentation, dependency pins, attribution, and integrity checks. Raw human records, participant-level reconstructed tables, manuscript files and internal review logs are not part of this repository.
