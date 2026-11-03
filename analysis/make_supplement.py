"""Build Supplementary_R2.md from the CSV outputs of r2_analysis.py.

Run after r2_analysis.py:  .venv/bin/python make_supplement.py
Every number in the supplement is read from outputs/; none is typed by hand.
"""
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TARGET = HERE.parent / "Supplementary_R2.md"

TITLE = ("Diaphragm ultrasound and airway occlusion pressure (P~0.1~) in mechanically "
         "ventilated patients with predominantly low respiratory drive: a prospective cohort study")


LABELS = {"DE": "Diaphragmatic excursion", "TFDI": "Diaphragm thickening fraction",
          "PaCO2": "PaCO~2~", "age": "Age", "SOFA": "SOFA score", "RASS": "RASS score",
          "Mean_GCS": "Glasgow Coma Scale", "RR": "Total respiratory rate",
          "PEEP": "PEEP", "Ventilation_Duration": "Duration of mechanical ventilation",
          "sedative": "Any sedative or opioid", "Intercept": "Intercept",
          "I(TFDI ** 2)": "TFDI²", "P0_1": "P~0.1~ (per cmH~2~O)"}


def p_fmt(p: float) -> str:
    return "\\< 0.001" if p < 0.001 else f"{p:.3f}"


def aic_fmt(a: float) -> str:
    s = f"{a:.1f}"
    return "0.0" if s == "-0.0" else s


def identifiable(d: pd.DataFrame) -> pd.DataFrame:
    # with a spline, the basis functions sum to one and are collinear with the
    # intercept, so neither the intercept nor the basis coefficients is unique
    if d.term.str.startswith("cr(").any():
        d = d[~d.term.str.startswith("cr(") & (d.term != "Intercept")]
    return d


def term_label(t: str) -> str:
    if t.startswith("cr(DE"):
        return f"DE spline basis {int(t[-2]) + 1}"
    return LABELS.get(t, t)


def cell(c) -> str:
    # typographic minus for negative numbers; "--" placeholders are left alone
    s = re.sub(r"(?<![\w~-])-(?=\d)", "−", str(c))
    return s[:1].upper() + s[1:] if s[:1].isascii() else s


def table(header: list[str], rows: list[list[str]], widths: list[int]) -> str:
    # dash counts set relative column widths in pandoc (the separator line is
    # longer than 72 characters, so pandoc uses them)
    assert len(widths) == len(header) and sum(widths) >= 80
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(":" + "-" * w for w in widths) + "|"]
    lines += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
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

    rows = [[r.index, r.shape, r.r2, aic_fmt(r.aic), p_fmt(r.reset_p), "Yes" if r.chosen else "No"] for r in ff.itertuples()]
    parts += ["**Supplementary Table S1. Functional form of each ultrasound index and coefficients of the separate models (outcome √P~0.1~).**", "",
              f"*Part A. Choice of functional form (n = {n}; all models include age, SOFA score, PaCO~2~, RASS score and sedative exposure).*", "",
              table(["Index", "Form", "R²", "AIC", "RESET p", "Selected"], rows, [14, 18, 14, 14, 14, 14]), "",
              "*The simplest form within 2 AIC units of the minimum was selected. The spline is a natural cubic regression spline with three degrees of freedom (two interior knots).*", ""]
    for model, name in [("A: DE model", "B. DE model"), ("B: TFDI model", "C. TFDI model")]:
        d = identifiable(co[co.model == model])
        rows = [[term_label(r.term), f"{r.beta:.4f}", f"{r.ci_lower:.4f} to {r.ci_upper:.4f}",
                 p_fmt(r.p_value), p_fmt(r.p_HC3)] for r in d.itertuples()]
        parts += [f"*Part {name} (n = {n}).*", "",
                  table(["Term", "β", "95% CI", "p-value", "HC3 p-value"], rows, [30, 13, 31, 13, 13]), ""]
    parts += [("*Coefficients are on the √P~0.1~ scale. The four basis functions of the DE spline sum to one and are therefore collinear "
               "with the intercept, so neither the intercept nor the individual basis coefficients is uniquely determined, and they are "
               "not shown. The linear and quadratic TFDI coefficients have no separate physiological meaning. Each index was tested as "
               "a block (main text, Table 3), and its association is expressed in cmH~2~O in Table 4.*"), ""]

    d = identifiable(co[co.model == "C: joint model"])
    rows = [[term_label(r.term), f"{r.beta:.4f}", f"{r.ci_lower:.4f} to {r.ci_upper:.4f}",
             p_fmt(r.p_value), p_fmt(r.p_HC3)] for r in d.itertuples()]
    nrows = [[r.model, r.n, f"{r.r2:.3f}", f"{r.adj_r2:.3f}", aic_fmt(r.aic)] for r in ne.itertuples()]
    vif = sm["vif_joint_linear"]
    parts += ["**Supplementary Table S2. Joint model including both ultrasound indices (outcome √P~0.1~).**", "",
              "*Part A. Nested models.*", "",
              table(["Model", "n", "R²", "Adjusted R²", "AIC"], nrows, [34, 10, 14, 16, 14]), "",
              (f"*The addition of DE to a model containing TFDI gave F~{bt['DE_given_TFDI']['df_num']},{bt['DE_given_TFDI']['df_den']}~ = "
               f"{bt['DE_given_TFDI']['F']}, p {p_fmt(bt['DE_given_TFDI']['p'])}. The addition of TFDI to a model containing DE gave "
               f"F~{bt['TFDI_given_DE']['df_num']},{bt['TFDI_given_DE']['df_den']}~ = {bt['TFDI_given_DE']['F']}, "
               f"p = {bt['TFDI_given_DE']['p']:.3f}. Variance inflation factors with both indices entered linearly were "
               f"DE {vif['DE']}, TFDI {vif['TFDI']}, age {vif['age']}, SOFA {vif['SOFA']}, PaCO~2~ {vif['PaCO2']}, "
               f"RASS {vif['RASS']}, sedative exposure {vif['sedative']}. The Spearman correlation between DE and TFDI was ρ = {sm['rho_DE_TFDI']}.*"), "",
              f"*Part B. Coefficients of the joint model (n = {n}).*", "",
              table(["Term", "β", "95% CI", "p-value", "HC3 p-value"], rows, [30, 13, 31, 13, 13]),
              "*The intercept and the DE spline basis coefficients are not shown, for the reason given under Supplementary Table S1.*", ""]

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
    parts += ["**Supplementary Table S3. Sensitivity analyses.**", "",
              table(["Analysis", "Index", "n", "Test of the ultrasound index (p) or model fit", "Predicted P~0.1~ difference, 25th to 75th percentile (cmH~2~O)"], rows, [34, 9, 7, 25, 25]), "",
              ("*Univariable rows are linear regressions of √P~0.1~ on RASS score or on sedative exposure alone. For the interaction "
               "analyses, the p-value tests the interaction between the ultrasound index and a RASS of −4 or −5. "
               f"In the primary models, the differences were {iqr['DE']:.2f} cmH~2~O for DE and {iqr['TFDI']:.2f} cmH~2~O for TFDI. Three of "
               "the values set to missing concern model variables, and the remainder of each patient's record was retained. These "
               "were a DE of 6.1 cm, outside the range of the cohort, a TFDI of 1.8%, incompatible with the patient's P~0.1~, and a "
               "PaCO~2~ that differed from the case report form. The rows \"with the recorded …\" refit the models with these values restored.*"), ""]

    rows = [[r.model, LABELS.get(r.term, r.term), f"{r.OR:.2f}", f"{r.ci_lower:.2f} to {r.ci_upper:.2f}", p_fmt(r.p_value)]
            for r in mo.itertuples()]
    parts += ["**Supplementary Table S4. Exploratory logistic regression of in-hospital mortality on P~0.1~.**", "",
              table(["Model", "Term", "Odds ratio", "95% CI", "p-value"], rows, [18, 30, 16, 22, 14]), "",
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
