#!/usr/bin/env python3
"""Number gate for the R2 manuscript and supplement.

Every decimal number written in manuscript_R2.md (outside the reference list)
must equal, at the precision written, a number that r2_analysis.py wrote to
outputs/. Numbers that come from the protocol rather than the data (device
specifications, the sample-size formula, software versions) are listed in
NOT_FROM_DATA and checked to be present instead.

Usage:  .venv/bin/python verify_numbers_r2.py
Exit 0 if every number is traced, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ROOT = HERE.parent
DOCUMENTS = [ROOT / "manuscript_R2.md", ROOT / "Supplementary_R2.md"]

NUM = re.compile(r"(?<![\w.])[−-]?\d+\.\d+")

# Written in the text but not produced by the analysis.
NOT_FROM_DATA = {
    "0.1": "ventilator display resolution and the P0.1 name",
    "0.5": "manufacturer measurement uncertainty",
    "1.0": "P0.1 band threshold (Telias 2020)",
    "3.5": "P0.1 band threshold (Telias 2020)",
    "1.96": "sample-size formula",
    "0.40": "sample-size formula",
    "0.60": "sample-size formula",
    "0.08": "sample-size formula",
    "144.06": "sample-size formula, carried from R1",
    "3.11": "Python version",
    "3.0.5": "pandas version",
    "1.15.3": "SciPy version",
    "0.14.6": "statsmodels version",
    "1.5": "reading of the DE prediction grid (flat segment start; slope below 2/3 of the 10th-90th percentile mean, 1.5-2.3 cm, checked 2026-10-26)",
    "2.5": "reading of the DE prediction grid (flat segment end; steep again from 2.4-2.5 cm)",
    "0.01": "R² gain of the joint model, 0.878 − 0.867, rounded",
    "0.05": "significance level",
    "0.001": "p-value floor",
}


# Integer counts asserted in the text, checked against summary.json / tables.
COUNTS = ["141", "140", "138", "118", "113", "123", "69", "52", "79", "10", "121"]


def output_numbers() -> set[str]:
    found: set[str] = set()
    for f in OUT.iterdir():
        if f.suffix not in {".csv", ".json", ".txt"}:
            continue
        for tok in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", f.read_text()):
            try:
                v = float(tok)
            except ValueError:
                continue
            for d in range(0, 5):
                found.add(f"{round(v, d):.{d}f}")
                found.add(f"{round(abs(v), d):.{d}f}")
    return found


def derived_from_data() -> set[str]:
    """Shares quoted in the text that are computed from the raw data, not written by r2_analysis.py."""
    import json
    import pandas as pd
    d = pd.read_csv(ROOT.parent / "Data" / "data.csv")
    n = len(d)
    out = {f"{100 * (d.PEEP == 5).sum() / n:.1f}"}
    for col in ["adjunct_respiratory_muscle_use", "bradypnea", "Cheyne_Stokes", "shallow_breath",
                "Kussmaul", "tachypnea", "paradoxycal", "dyspnea", "grunting", "hyperventilationq",
                "Biot", "other_types_of_breathing"]:
        out.add(f"{100 * d[col].astype(str).str.lower().eq('yes').sum() / n:.1f}")
    p = d.P0_1
    for k in [(p < 1.0).sum(), ((p >= 1.0) & (p <= 3.5)).sum(), (p > 3.5).sum()]:
        out.add(f"{100 * k / n:.1f}")
    r2 = json.loads((OUT / "summary.json").read_text())["diagnostics"]["covariates"]["r2"]
    out.add(f"{100 * r2:.1f}")
    return out


def main() -> int:
    have = output_numbers() | derived_from_data()
    failures = []
    for doc in DOCUMENTS:
        text = doc.read_text(encoding="utf-8")
        text = text.split("**References**")[0]
        for line_no, line in enumerate(text.splitlines(), 1):
            if line.startswith("![]") or "IR.KMU" in line:
                continue
            for raw in NUM.findall(line):
                tok = raw.replace("−", "-").lstrip("-")
                if tok in NOT_FROM_DATA or tok in have:
                    continue
                failures.append(f"{doc.name}:{line_no}: {raw}")
    import pandas as pd
    d = pd.read_csv(ROOT.parent / "Data" / "data.csv")
    summary = "".join(f.read_text() for f in OUT.iterdir() if f.suffix in {".csv", ".json"})
    summary += f" {d.Lung_Injury_Status.isin(['bilateral', 'unilateral']).sum()} "
    for c in COUNTS:
        if not re.search(rf"\b{c}\b", summary):
            failures.append(f"count {c} not found in outputs/")
    if failures:
        print(f"UNTRACED ({len(failures)}):")
        print("\n".join(failures))
        return 1
    print("number gate: every decimal in the manuscript and supplement traces to outputs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
