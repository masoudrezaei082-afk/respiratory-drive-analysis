# Respiratory-drive statistical analysis

Analysis code for *Diaphragm ultrasound and airway occlusion pressure (P0.1) in mechanically
ventilated patients with predominantly low respiratory drive: a prospective cohort study*
(Raeesi Estabragh et al., submitted to *Scientific Reports*).

The code runs against private patient-level files supplied outside the repository. Patient data,
generated results, manuscripts and review material are intentionally excluded, and rows are
referred to by their position in the data file, never by a hospital identifier.

## Contents

| File | Purpose |
|---|---|
| `analysis/r2_analysis.py` | Second-revision analysis: separate DE and TFDI models for square-root P0.1 with pre-specified covariates (age, SOFA, PaCO2, RASS, sedative exposure), HC3 standard errors, back-transformed predictions with bootstrap intervals, sedation and other sensitivity analyses, exploratory mortality model. Writes `analysis/outputs/`. |
| `analysis/revision_analysis.py` | First-revision analysis. Its loader and cohort checks are reused by `r2_analysis.py`. |
| `analysis/fix_data_audit.py` | Applies the four documented data-preparation corrections to `data.csv` from the original spreadsheet, keeping an untouched copy. |
| `analysis/make_figure1.py` | Draws the patient-flow figure from the analysis outputs (needs Graphviz `dot`). |
| `analysis/make_supplement.py` | Builds the supplementary tables from the analysis outputs. |
| `analysis/verify_numbers_r2.py` | Checks that every number in the manuscript and supplement traces to an output (needs the manuscript files, which are not public). |
| `analysis/verify_outputs.py` | Checks the first-revision output set. |

## Run

Python 3.11, with the pinned packages:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

export RESPDRIVE_DATA=/private/path/data.csv
python analysis/r2_analysis.py
python analysis/make_figure1.py
python analysis/make_supplement.py
```

`r2_analysis.py` expects the 141-patient cohort file and stops before fitting if its checks fail.
The bootstrap seed is fixed (20261026, 2000 resamples), so reruns give identical outputs.
`fix_data_audit.py` reads `data.csv` and `original_data.xlsx` from `RESPDRIVE_DATA_DIR`.

The first-revision analysis runs as before:

```bash
python analysis/revision_analysis.py --input /private/path/data.csv --output-dir outputs
```

## Repository boundary

Only analysis code and metadata are tracked. Do not add patient data, tabular or statistical data
files, generated outputs, figures, documents, manuscripts, reviewer correspondence, logs, or
scheduler files.

## License

MIT; see `LICENSE`.
