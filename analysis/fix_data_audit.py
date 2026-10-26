#!/usr/bin/env python3
"""Apply the four preparation fixes found by the audit (R2/data_audit_R2.md) to Data/data.csv.

The fixes are derived from Dr Raeisi's original spreadsheet (Data/original_data.xlsx),
whose rows are in the same order as data.csv:

  D1  DE in data row 122 restored to the recorded 6.1 cm. r2_analysis.py excludes it
      from the models as implausible, with the reason logged.
  D2  Breathing-pattern flags rebuilt from the original codes as whole numbers.
      Before, "10", "11" and "12" also set code 1 (accessory muscles) and "12"
      also set code 2 (grunting).
  D3  "Any comorbidity" set to yes when a specific comorbidity is yes. Two
      patients carry code 5 ("none") together with a disease code.
  D4  FiO2 left blank where the original records no FiO2, rather than "0%".

The first run copies data.csv to data_pre_audit.csv. Every run starts from that
copy, so the script can be re-run and gives the same file.

    python3 analysis/fix_data_audit.py        (run from R2/)

RESPDRIVE_DATA_DIR points at the private folder holding data.csv and
original_data.xlsx. Rows are referred to by position (counting from 1 below the
header), never by hospital identifier.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(os.environ.get("RESPDRIVE_DATA_DIR", Path(__file__).resolve().parents[2] / "Data"))
CSV = DATA_DIR / "data.csv"
BACKUP = DATA_DIR / "data_pre_audit.csv"
ORIGINAL = DATA_DIR / "original_data.xlsx"

# CRF codes for the breathing pattern (نوع تنفس), read from the form in Data/1.jpg.
BREATHING = {1: "adjunct_respiratory_muscle_use", 2: "grunting", 3: "shallow_breath", 4: "tachypnea",
             5: "bradypnea", 6: "dyspnea", 7: "paradoxycal", 8: "Cheyne_Stokes", 9: "hyperventilationq",
             10: "Kussmaul", 11: "Biot", 12: "other_types_of_breathing"}
COMORBIDITIES = ["CAD", "pulmonary_dz", "HTN", "renal", "other_PMH"]


def check(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"fix_data_audit: {message}")


def codes(cell) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", str(cell))} if pd.notna(cell) else set()


def write(d: pd.DataFrame) -> bytes:
    """data.csv as it was saved: BOM, CRLF, and the second PaO2 header without pandas' ".1"."""
    text = d.to_csv(index=False, lineterminator="\r\n")
    header, rest = text.split("\r\n", 1)
    return (header.replace("PaO2.1", "PaO2") + "\r\n" + rest).encode("utf-8-sig")


def main() -> int:
    if not BACKUP.exists():
        shutil.copy2(CSV, BACKUP)
        print(f"backup written: {BACKUP.name}")
    raw = BACKUP.read_bytes()
    d = pd.read_csv(BACKUP, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    check(write(d) == raw, "data.csv does not round-trip unchanged; refusing to rewrite it")
    o = pd.read_excel(ORIGINAL, dtype=str)
    ids_o = o.iloc[:, 0].str.replace("`", "").str.strip()
    ids_d = d["admission_id"].str.replace("`", "").str.strip()
    check(len(o) == len(d) and (ids_o.values == ids_d.values).all(), "rows of original and data.csv do not align")
    log = []

    # D1
    r = 122 - 1
    check(d.loc[r, "DE"] == "" and float(o.loc[r, "DE"]) == 6.1,
          "D1: expected a blank DE in data row 122 and 6.1 in the original")
    d.loc[r, "DE"] = "6.1"
    log.append("D1  DE 6.1 restored in data row 122")

    # D2
    breathing = o["نوع تنفس"].map(codes)
    for code, col in BREATHING.items():
        new = breathing.map(lambda s: "Yes" if code in s else "No")
        changed = int((new != d[col]).sum())
        d[col] = new
        if changed:
            log.append(f"D2  {col}: {changed} row(s) corrected, now {int((new == 'Yes').sum())} yes")
    check((d["adjunct_respiratory_muscle_use"] == "Yes").sum() == 54 and (d["grunting"] == "Yes").sum() == 2,
          "D2: expected 54 accessory-muscle and 2 grunting patients")

    # D3
    any_dz = d[COMORBIDITIES].eq("yes").any(axis=1)
    fix = any_dz & d["comorbitity"].eq("no")
    check(fix.sum() == 2, f"D3: expected 2 contradictory rows, found {int(fix.sum())}")
    d.loc[fix, "comorbitity"] = "yes"
    log.append(f"D3  comorbidity set to yes in data rows {', '.join(str(i + 1) for i in d.index[fix])}")

    # D4
    zero = d["FiO2"].eq("0%")
    check(zero.sum() == 2 and o.loc[zero, "P/F RATIO"].str.contains("/").sum() == 0,
          "D4: expected two FiO2 0% rows with no FiO2 in the original")
    d.loc[zero, "FiO2"] = ""
    log.append(f"D4  FiO2 0% blanked in data rows {', '.join(str(i + 1) for i in d.index[zero])}")

    CSV.write_bytes(write(d))
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
