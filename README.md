# Respiratory-drive statistical analysis

This repository contains the analysis code and dependency metadata for a respiratory-drive cohort study. It runs against a private CSV supplied outside the repository. Patient-level data, generated results, manuscripts, and review material are intentionally excluded.

## Requirements

- Python 3.11 (tested interpreter family)
- A private CSV containing the required columns:
  `FiO2`, `Sedation_Depth`, `Lung_Injury_Status`, `outcome`, `P0_1`, `DE`,
  `TFDI`, `PaCO2`, `age`, `SOFA`, `VR`, `PH`, `morphine`, `P_F_ratio`,
  `SPO2`, `SaO2`, and `Admission_Diagnosis`.

The analysis checks the supplied cohort for 141 rows and P0.1 quartiles of 0.60, 1.30, and 1.80. It fails before model fitting when required columns or cohort checks do not match.

## Run

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/revision_analysis.py \
  --input /private/path/data.csv \
  --output-dir outputs
`--expected-md5 HEX` is optional when a private frozen input needs an external checksum check. The analysis writes 35 artifacts, including model tables, diagnostics, summaries, and figures. `outputs/` is ignored by Git; choose a different output path if results must remain outside the working tree.
```

`--expected-md5 HEX` is optional when a private frozen input needs an external checksum check. The analysis writes 34 artifacts, including model tables, diagnostics, summaries, and figures. `outputs/` is ignored by Git; choose a different output path if results must remain outside the working tree.

## Repository boundary

Only analysis code and metadata are tracked. Do not add patient data, tabular or statistical data files, generated outputs, figures, documents, manuscripts, reviewer correspondence, logs, or scheduler files.
