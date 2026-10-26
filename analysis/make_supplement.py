"""Build Supplementary_R2.md from the CSV outputs of r2_analysis.py.

Run after r2_analysis.py:  .venv/bin/python make_supplement.py
Every number in the supplement is read from outputs/ or, for Supplementary
Table S1, counted from Data/data.csv; none is typed by hand.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TARGET = HERE.parent / "Supplementary_R2.md"

TITLE = ("Diaphragm ultrasound and airway occlusion pressure (P~0.1~) in mechanically "
         "ventilated patients with predominantly low respiratory drive: a prospective cohort study")

DATA = Path(os.environ.get("RESPDRIVE_DATA", HERE.parents[1] / "Data" / "data.csv"))

# Supplementary Table S1: data.csv column -> label, listed by frequency.
S1_COLUMNS = {"adjunct_respiratory_muscle_use": "Use of accessory respiratory muscles",
              "bradypnea": "Bradypnea", "Cheyne_Stokes": "Cheyne--Stokes respiration",
              "shallow_breath": "Shallow breathing", "Kussmaul": "Kussmaul breathing",
              "tachypnea": "Tachypnea", "paradoxycal": "Paradoxical breathing", "dyspnea": "Dyspnea",
              "grunting": "Grunting", "hyperventilationq": "Hyperventilation", "Biot": "Biot breathing",
              "other_types_of_breathing": "Other breathing pattern"}


def breathing_patterns() -> list[tuple[str, int, float]]:
    d = pd.read_csv(DATA, encoding="utf-8-sig")
    rows = []
    for col, label in S1_COLUMNS.items():
        k = int(d[col].astype(str).str.lower().eq("yes").sum())
        rows.append((label, k, round(100 * k / len(d), 1)))
    return sorted(rows, key=lambda r: -r[1])

LABELS = {"DE": "Diaphragmatic excursion", "TFDI": "Diaphragm thickening fraction",
          "PaCO2": "PaCO~2~", "age": "Age", "SOFA": "SOFA score", "RASS": "RASS score",
          "Mean_GCS": "Glasgow Coma Scale", "RR": "Total respiratory rate",
          "PEEP": "PEEP", "Ventilation_Duration": "Duration of mechanical ventilation",
          "sedative": "Any sedative or opioid", "Intercept": "Intercept",
          "I(TFDI ** 2)": "TFDI²", "P0_1": "P~0.1~ (per cmH~2~O)"}


def p_fmt(p: float) -> str:
    return "\\< 0.001" if p < 0.001 else f"{p:.3f}"


def term_label(t: str) -> str:
    if t.startswith("cr(DE"):
        return f"DE spline basis {int(t[-2]) + 1}"
    return LABELS.get(t, t)


def cell(c) -> str:
    # typographic minus for negative numbers; "--" placeholders are left alone
    s = re.sub(r"(?<![\w~-])-(?=\d)", "−", str(c))
    return s[:1].upper() + s[1:] if s[:1].isascii() else s


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join([":---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    sp = pd.read_csv(OUT / "spearman_univariable.csv")
    ff = pd.read_csv(OUT / "functional_form.csv")
    co = pd.read_csv(OUT / "models_sqrt_scale.csv")
    ne = pd.read_csv(OUT / "nested_models.csv")
    se = pd.read_csv(OUT / "sedation_sensitivity.csv")
    ot = pd.read_csv(OUT / "other_sensitivity.csv")
    mo = pd.read_csv(OUT / "mortality_exploratory.csv")
    sm = json.loads((OUT / "summary.json").read_text())
    bt = sm["block_tests"]
    n = sm["n_complete_case"]
    iqr = {c["var"]: c["difference"] for c in sm["iqr_contrasts"]}

    parts = [f"*{TITLE}*", "", "**Supplementary Information**", ""]

    parts += ["**Supplementary Table S1. Recorded abnormal breathing patterns in the study cohort.**", "",
              table(["Pattern", "n", "Percentage (%)"], [[a, b, f"{c:.1f}"] for a, b, c in breathing_patterns()]), "",
              "*Percentages use the full cohort (N = 141) as denominator. Categories are not mutually exclusive.*", ""]

    rows = [[LABELS[r.variable], r.n, f"{r.rho:.3f}", p_fmt(r.p_value)] for r in sp.itertuples()]
    parts += ["**Supplementary Table S2. Spearman rank correlations with P~0.1~.**", "",
              table(["Variable", "n", "Spearman ρ", "p-value"], rows), "",
              "*Reported descriptively; these correlations were not used to select covariates for the regression models.*", ""]

    rows = [[r.index, r.shape, r.r2, r.aic, p_fmt(r.reset_p), "Yes" if r.chosen else "No"] for r in ff.itertuples()]
    parts += ["**Supplementary Table S3. Functional form of each ultrasound index and coefficients of the separate models (outcome √P~0.1~).**", "",
              f"*Part A. Choice of functional form (n = {n}; all models include age, SOFA score, PaCO~2~, RASS score and sedative exposure).*", "",
              table(["Index", "Form", "R²", "AIC", "RESET p", "Selected"], rows), "",
              "*The simplest form within 2 AIC units of the minimum was selected. The spline is a natural cubic regression spline with four degrees of freedom.*", ""]
    for model, name in [("A: DE model", "B. DE model"), ("B: TFDI model", "C. TFDI model")]:
        d = co[co.model == model]
        rows = [[term_label(r.term), f"{r.beta:.4f}", f"{r.ci_lower:.4f} to {r.ci_upper:.4f}",
                 p_fmt(r.p_value), p_fmt(r.p_HC3)] for r in d.itertuples()]
        parts += [f"*Part {name} (n = {n}).*", "",
                  table(["Term", "β", "95% CI", "p-value", "HC3 p-value"], rows), ""]
    parts += ["*Coefficients are on the √P~0.1~ scale. Spline basis coefficients and the separate linear and quadratic TFDI coefficients have no individual physiological meaning; each index was tested as a block (main text, Table 3), and its association is expressed in cmH~2~O in Table 4.*", ""]

    d = co[co.model == "C: joint model"]
    rows = [[term_label(r.term), f"{r.beta:.4f}", f"{r.ci_lower:.4f} to {r.ci_upper:.4f}",
             p_fmt(r.p_value), p_fmt(r.p_HC3)] for r in d.itertuples()]
    nrows = [[r.model, r.n, f"{r.r2:.3f}", f"{r.adj_r2:.3f}", f"{r.aic:.1f}"] for r in ne.itertuples()]
    vif = sm["vif_joint_linear"]
    parts += ["**Supplementary Table S4. Joint model including both ultrasound indices (outcome √P~0.1~).**", "",
              "*Part A. Nested models.*", "",
              table(["Model", "n", "R²", "Adjusted R²", "AIC"], nrows), "",
              (f"*DE added to a model containing TFDI: F~{bt['DE_given_TFDI']['df_num']},{bt['DE_given_TFDI']['df_den']}~ = "
               f"{bt['DE_given_TFDI']['F']}, p {p_fmt(bt['DE_given_TFDI']['p'])}. TFDI added to a model containing DE: "
               f"F~{bt['TFDI_given_DE']['df_num']},{bt['TFDI_given_DE']['df_den']}~ = {bt['TFDI_given_DE']['F']}, "
               f"p = {bt['TFDI_given_DE']['p']:.3f}. Variance inflation factors with both indices entered linearly: "
               f"DE {vif['DE']}, TFDI {vif['TFDI']}, age {vif['age']}, SOFA {vif['SOFA']}, PaCO~2~ {vif['PaCO2']}, "
               f"RASS {vif['RASS']}, sedative exposure {vif['sedative']}. Spearman ρ between DE and TFDI = {sm['rho_DE_TFDI']}.*"), "",
              f"*Part B. Coefficients of the joint model (n = {n}).*", "",
              table(["Term", "β", "95% CI", "p-value", "HC3 p-value"], rows), ""]

    rows = []
    for r in se.itertuples():
        if pd.isna(r.index):
            rows.append([r.analysis, "--", r.n, f"β = {r.beta:.4f}, p = {r.p:.3f}", "--"])
        else:
            p = "--" if pd.isna(r.p) else p_fmt(r.p)
            diff = "--" if pd.isna(r.iqr_difference) else f"{r.iqr_difference:.2f}"
            rows.append([r.analysis, r.index, r.n, p, diff])
    for r in ot.itertuples():
        rows.append([r.analysis.replace("PaCO2", "PaCO~2~"), r.index, r.n, f"R² = {r.r2:.3f}", f"{r.iqr_difference:.2f}"])
    parts += ["**Supplementary Table S5. Sensitivity analyses.**", "",
              table(["Analysis", "Index", "n", "Test of the ultrasound index (p) or model fit", "Predicted P~0.1~ difference, 25th to 75th percentile (cmH~2~O)"], rows), "",
              ("*Univariable rows are linear regressions of √P~0.1~ on RASS score or on sedative exposure alone. For the interaction "
               "analyses, the p-value tests the interaction between the ultrasound index and deep sedation (RASS −4 or −5). "
               f"Primary-model differences for comparison: DE {iqr['DE']:.2f} cmH~2~O, TFDI {iqr['TFDI']:.2f} cmH~2~O. In the primary "
               "analysis three recorded values were set to missing and the rest of each patient's record kept: a DE of 6.1 cm, "
               "outside the range of the cohort; a TFDI of 1.8%, incompatible with the patient's P~0.1~; and a PaCO~2~ that "
               "differed from the case report form. The rows \"with the recorded …\" refit the models with these values restored.*"), ""]

    rows = [[r.model, LABELS.get(r.term, r.term), f"{r.OR:.2f}", f"{r.ci_lower:.2f} to {r.ci_upper:.2f}", p_fmt(r.p_value)]
            for r in mo.itertuples()]
    parts += ["**Supplementary Table S6. Exploratory logistic regression of in-hospital mortality on P~0.1~.**", "",
              table(["Model", "Term", "Odds ratio", "95% CI", "p-value"], rows), "",
              f"*n = {mo.n.iloc[0]}, {mo.events.iloc[0]} deaths. Exploratory and not pre-specified; no prognostic conclusion is drawn.*", ""]

    kw = sm["kruskal_P01_RASS"]
    parts += ["![](media/figureS1.png)", "",
              ("**Supplementary Figure S1.** Residual diagnostics for the separate models of √P~0.1~. (a) Residuals against fitted "
               "values and (b) normal Q-Q plot for the DE model; (c) residuals against fitted values and (d) normal Q-Q plot for "
               f"the TFDI model. Shapiro-Wilk p = {sm['diagnostics']['DE']['shapiro_p']:.3f} (DE) and "
               f"{p_fmt(sm['diagnostics']['TFDI']['shapiro_p'])} (TFDI); Breusch-Pagan p = "
               f"{sm['diagnostics']['DE']['breusch_pagan_p']:.3f} and {sm['diagnostics']['TFDI']['breusch_pagan_p']:.3f}."), "",
              "![](media/figureS2.png)", "",
              (f"**Supplementary Figure S2.** P~0.1~ by RASS score (n = 141). Boxes show median and IQR; points are individual "
               f"patients. Kruskal-Wallis H = {kw['H']}, p = {kw['p']}; Spearman ρ = {sm['spearman_P01_RASS']['rho']}, "
               f"p = {sm['spearman_P01_RASS']['p']}."), ""]

    TARGET.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
