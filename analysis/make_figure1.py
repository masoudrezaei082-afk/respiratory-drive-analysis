#!/usr/bin/env python3
"""Draw Figure 1 (patient flow) with Graphviz from the counts in outputs/summary.json.

Replaces the hand-made figure1.jpg, which gave the excluded patient's reason as
"missing DE measurement"; the value was recorded (6.1 cm) and set aside as
implausible (data_audit_R2.md, D1). Since then a TFDI and a PaCO2 have also been
set to missing (r2_analysis.py, IMPLAUSIBLE); those patients stay in Tables 1-2.

Run after r2_analysis.py:  python3 analysis/make_figure1.py   (from R2/)
"""
import csv
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TARGET = HERE.parent / "media" / "figure1.png"


def main() -> None:
    s = json.loads((OUT / "summary.json").read_text())
    n_all = 141
    n_cc = s["n_complete_case"]
    with open(OUT / "other_sensitivity.csv") as fh:
        n_li = int(next(r["n"] for r in csv.DictReader(fh) if r["analysis"] == "adding lung-injury laterality"))
    de_max = s["DE_max_retained"]
    box = 'shape=box, style="rounded,filled", fillcolor="#E8EEF5", color="#44546A", penwidth=1.6, fontname="Helvetica", fontsize=13, margin="0.25,0.12"'
    dot = f"""digraph flow {{
  graph [rankdir=TB, nodesep=0.6, ranksep=0.45, dpi=300, bgcolor=white, splines=ortho];
  node [{box}];
  edge [color="#44546A", penwidth=1.6, arrowsize=0.8];
  enrolled [label=<<b>Patients screened and enrolled (n = {n_all})</b><br/>Invasive mechanical ventilation ≤ 7 days,<br/>PaO<sub>2</sub>/FiO<sub>2</sub> ≤ 300>];
  excl1 [label=<<b>Excluded from primary models (n = {n_all - n_cc})</b><br/>One model variable recorded implausibly<br/>and set to missing:<br/>DE 6.1 cm (cohort maximum {de_max} cm), n = 1<br/>TFDI 1.8%, n = 1<br/>PaCO<sub>2</sub> not matching the case report form, n = 1>];
  primary [label=<<b>Primary multivariable analysis cohort</b><br/>(n = {n_cc})>];
  model1 [label=<<b>Primary multivariable models (n = {n_cc})</b><br/>Without lung-injury laterality>];
  excl2 [label=<<b>Excluded from sensitivity model (n = {n_cc - n_li})</b><br/>Lung-injury laterality not recorded>];
  model2 [label=<<b>Sensitivity analysis model (n = {n_li})</b><br/>Complete cases with laterality included>];
  {{rank=same; enrolled_pt [shape=point, width=0, height=0, label=""]; }}
  enrolled -> enrolled_pt [arrowhead=none];
  enrolled_pt -> primary;
  enrolled_pt -> excl1;
  {{rank=same; enrolled_pt; excl1}}
  primary -> model1;
  primary -> excl2;
  excl2 -> model2;
  {{rank=same; model1; excl2}}
}}"""
    TARGET.parent.mkdir(exist_ok=True)
    subprocess.run(["dot", "-Tpng", "-o", str(TARGET)], input=dot.encode(), check=True)
    print(f"written {TARGET}")


if __name__ == "__main__":
    main()
