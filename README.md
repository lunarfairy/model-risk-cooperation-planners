# Model risk in cooperation planners

Code and simulation data for **Model risk in AI planners for human cooperation: budgets, local information, privacy and scale**.

## Run

Use Python 3.12 and install the recorded dependencies:

```text
python -m pip install -r requirements-lock.txt
python smoke_test.py data/calib_v2.json
python figures.py data work/figures
```

The figure command generates the five main figures, three supplementary figures, and numerical tables from the included results.

## Contents

- `engine2.py`: game simulator and planners.
- `calib2.py`: behavioural-model calibration.
- `runs2.py`: experiment driver.
- `figures.py`: figures and summary tables.
- `data/`: calibration outputs and simulation data.

Full workflow and experiment settings: [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Data provenance: [DATA_SOURCES.md](DATA_SOURCES.md). File checksums: [SHA256SUMS](SHA256SUMS).

Code: [MIT License](LICENSE). Data: [CC BY 4.0](data/LICENSE.md).
