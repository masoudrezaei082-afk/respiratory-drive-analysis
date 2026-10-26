#!/usr/bin/env python3
"""Round-2 re-analysis: separate ultrasound models, clinically chosen covariates,
back-transformed predictions, sedation sensitivity and exploratory mortality.

Reuses the R1 loader (cohort checks, FiO2 parsing, RASS range) from
revision_analysis.py, which sits beside this file in the code archive.

The patient-level data are not public. Point RESPDRIVE_DATA at the private
data.csv, then run with the pinned environment (requirements.txt):
    RESPDRIVE_DATA=/private/path/data.csv python analysis/r2_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # .../Scientific Report in the working tree
sys.path[:0] = [str(HERE), str(ROOT / "01_revision" / "code_repository" / "analysis")]
from revision_analysis import load, require  # noqa: E402

DATA = Path(os.environ.get("RESPDRIVE_DATA", ROOT / "Data" / "data.csv"))
OUT = HERE / "outputs"
SEED = 20261026
N_BOOT = 2000

# Covariates chosen on clinical grounds before fitting (R4-2, R3-2): each is a
# recognised determinant of drive or of the occlusion pressure, entered
# regardless of its univariable p-value.
COVARIATES = ["age", "SOFA", "PaCO2", "RASS", "sedative"]
COV = " + ".join(COVARIATES)

# Candidate shapes for each index. The one with the lowest AIC is used, the
# simpler one when AICs are within 2; splines are capped at 4 df for n = 138.
SHAPES = {
    "DE": {"linear": "DE", "quadratic": "DE + I(DE**2)", "spline": "cr(DE, df=4)"},
    "TFDI": {"linear": "TFDI", "quadratic": "TFDI + I(TFDI**2)", "spline": "cr(TFDI, df=4)"},
}
ORDER = ["linear", "quadratic", "spline"]

# Recorded values that are impossible or contradicted by the source form. Each is
# set to missing; the rest of the patient's record is kept (report.md, section 6).
# The models are refitted with them restored in other_sensitivity.csv. Keys are
# (data row in data.csv, counting from 1 below the header; column), not hospital
# identifiers, so the code can be published.
IMPLAUSIBLE = {
    (122, "DE"): 6.1,       # 2 cm above every other patient (maximum 4.1); P0.1 0.7, TFDI 21%
    (19, "TFDI"): 1.8,      # no thickening at all, against a P0.1 of 1.3 and a DE of 1.8 cm
    (1, "PaCO2"): 66.0,     # the case report form records 100.6 and computes VR from it
    (59, "PH"): 1.17,       # incompatible with life
    (80, "VEP"): 48.0,      # VEP is PBW x 0.1 L/min; 48 is ten times the expected 4.8
    (127, "VEP"): 0.4,      # expected 6.7 from PBW
    (136, "VR"): 43.0,      # would need a minute ventilation near 285 L/min; next highest 4.6
}
RESTORED = ["DE", "TFDI", "PaCO2"]


def prepare() -> pd.DataFrame:
    df = load(DATA)
    for col in RESTORED:
        df[f"{col}_recorded"] = df[col]
    require(df.index.equals(pd.RangeIndex(len(df))), "loader must keep the file's row order")
    for (row, col), value in IMPLAUSIBLE.items():
        hit = (df.index == row - 1) & (df[col] == value)
        require(hit.sum() == 1, f"expected {col} {value} in data row {row}")
        df.loc[hit, col] = np.nan
        print(f"{col} {value} in data row {row} set to missing as implausible")
    # Two PaO2 values per patient: one written inside the P/F field, one in its
    # own column. Dr Raeisi (R2): the higher is the arterial gas, and P/F is that
    # PaO2 over FiO2. Not reported while B1 is open (report.md).
    df["PaO2_arterial"] = df[["PaO2", "PaO2.1"]].max(axis=1)
    df["PF_arterial"] = df["PaO2_arterial"] / df["FiO2_frac"]
    agents = ["midazolam", "morphine", "methadone", "fendanyl"]
    df["sedative"] = df[agents].eq("yes").any(axis=1).astype(int)
    df["sqrtP"] = np.sqrt(df["P0_1"])
    df["deep"] = (df["RASS"] <= -4).astype(int)
    df["RASS_cat"] = pd.cut(df["RASS"], [-6, -5, -4, -3, 5],
                            labels=["-5", "-4", "-3", "-2 or lighter"]).astype(str)
    return df


def checks(res) -> dict:
    return {
        "n": int(res.nobs),
        "r2": round(float(res.rsquared), 4),
        "adj_r2": round(float(res.rsquared_adj), 4),
        "aic": round(float(res.aic), 1),
        "shapiro_p": round(float(stats.shapiro(res.resid)[1]), 4),
        "breusch_pagan_p": round(float(het_breuschpagan(res.resid, res.model.exog)[1]), 4),
        "reset_p": round(float(linear_reset(res, power=2, use_f=True).pvalue), 4),
    }


def block_test(full, reduced) -> dict:
    """F-test for the terms in full that are absent from reduced (same rows)."""
    f, p, df_diff = full.compare_f_test(reduced)
    return {"F": round(float(f), 1), "df_num": int(df_diff),
            "df_den": int(full.df_resid), "p": float(p)}


def base_values(data: pd.DataFrame) -> dict:
    base = {c: float(data[c].median()) for c in ["age", "SOFA", "PaCO2", "RASS", "DE", "TFDI"]}
    base["sedative"] = int(data["sedative"].mode()[0])
    if "RASS_cat" in data:
        base["RASS_cat"] = data["RASS_cat"].mode()[0]
    for c in ["deep", "bilateral"]:
        if c in data:
            base[c] = int(data[c].mode()[0])
    return base


def predict_grid(res, data: pd.DataFrame, var: str, values) -> pd.DataFrame:
    """Predicted P0.1 (cmH2O) at given values of var, other covariates at the
    median (sedative at its mode). Squaring the fitted sqrt-scale mean and its
    CI gives the conditional median of P0.1, since sqrt(P0.1) errors are symmetric."""
    base = base_values(data)
    grid = pd.DataFrame([{**base, var: v} for v in values])
    pr = res.get_prediction(grid).summary_frame(alpha=0.05)
    return pd.DataFrame({
        "value": np.round(values, 2),
        "pred_P01": (pr["mean"] ** 2).round(2).values,
        "ci_lower": (pr["mean_ci_lower"].clip(lower=0) ** 2).round(2).values,
        "ci_upper": (pr["mean_ci_upper"] ** 2).round(2).values,
    })


def iqr_point(res, data: pd.DataFrame, var: str) -> float:
    q25, q75 = data[var].quantile([0.25, 0.75])
    g = predict_grid(res, data, var, np.array([q25, q75]))
    return round(float(g["pred_P01"].iloc[1] - g["pred_P01"].iloc[0]), 2)


def iqr_contrast(formula: str, data: pd.DataFrame, var: str, rng) -> dict:
    """Difference in predicted P0.1 between the 75th and 25th percentile of var,
    with a percentile bootstrap CI (the delta method is awkward after squaring).
    Percentiles are held at the full-sample values in every resample."""
    q25, q75 = data[var].quantile([0.25, 0.75])

    def contrast(d: pd.DataFrame) -> float:
        r = smf.ols(formula, data=d).fit()
        g = predict_grid(r, d, var, np.array([q25, q75]))
        return float(g["pred_P01"].iloc[1] - g["pred_P01"].iloc[0])

    point = contrast(data)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(data), len(data))
        boots.append(contrast(data.iloc[idx].reset_index(drop=True)))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"var": var, "p25": round(float(q25), 2), "p75": round(float(q75), 2),
            "difference": round(point, 2), "ci_lower": round(float(lo), 2),
            "ci_upper": round(float(hi), 2), "n_boot": N_BOOT}


def choose_shape(cc: pd.DataFrame, var: str) -> tuple[str, pd.DataFrame]:
    rows = []
    for name in ORDER:
        r = smf.ols(f"sqrtP ~ {SHAPES[var][name]} + {COV}", data=cc).fit()
        rows.append({"index": var, "shape": name, "term": SHAPES[var][name], **checks(r)})
    tab = pd.DataFrame(rows)
    best = tab["aic"].min()
    chosen = next(n for n in ORDER                  # simplest shape within 2 AIC of the best
                  if tab.loc[tab["shape"] == n, "aic"].iloc[0] - best < 2)
    tab["chosen"] = tab["shape"] == chosen
    return SHAPES[var][chosen], tab


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = prepare()
    summary: dict = {}

    cc = df[["sqrtP", "P0_1", "DE", "TFDI", "RASS_cat", "deep"] + COVARIATES].dropna().copy()
    require(len(cc) == 138, f"expected complete-case n=138, got {len(cc)}")
    summary["n_complete_case"] = len(cc)
    summary["n_sedative"] = int(df["sedative"].sum())
    summary["n_deep_rass"] = int(df["deep"].sum())
    summary["n_rass_minus5"] = int((df["RASS"] == -5).sum())

    de_term, de_tab = choose_shape(cc, "DE")
    tf_term, tf_tab = choose_shape(cc, "TFDI")
    pd.concat([de_tab, tf_tab]).to_csv(OUT / "functional_form.csv", index=False)
    summary["shape_DE"] = de_term
    summary["shape_TFDI"] = tf_term

    f_de = f"sqrtP ~ {de_term} + {COV}"
    f_tf = f"sqrtP ~ {tf_term} + {COV}"
    f_joint = f"sqrtP ~ {de_term} + {tf_term} + {COV}"
    f_cov = f"sqrtP ~ {COV}"
    m_de = smf.ols(f_de, data=cc).fit()
    m_tf = smf.ols(f_tf, data=cc).fit()
    m_joint = smf.ols(f_joint, data=cc).fit()
    m_cov = smf.ols(f_cov, data=cc).fit()

    # Coefficients on the sqrt scale (Supplementary), classical and HC3 p-values.
    rows = []
    for lab, m in [("A: DE model", m_de), ("B: TFDI model", m_tf),
                   ("C: joint model", m_joint), ("Covariates only", m_cov)]:
        ci, hc3 = m.conf_int(), m.model.fit(cov_type="HC3")
        for t in m.params.index:
            rows.append({"model": lab, "n": int(m.nobs), "term": t,
                         "beta": round(m.params[t], 4), "ci_lower": round(ci.loc[t, 0], 4),
                         "ci_upper": round(ci.loc[t, 1], 4), "p_value": round(m.pvalues[t], 4),
                         "p_HC3": round(hc3.pvalues[t], 4)})
    pd.DataFrame(rows).to_csv(OUT / "models_sqrt_scale.csv", index=False)

    diag = {lab: checks(m) for m, lab in [(m_de, "DE"), (m_tf, "TFDI"),
                                          (m_joint, "joint"), (m_cov, "covariates")]}
    for lab, m in [("DE", m_de), ("TFDI", m_tf), ("joint", m_joint)]:
        diag[lab]["cook_n_flagged"] = int((m.get_influence().cooks_distance[0] > 4 / len(cc)).sum())
    summary["diagnostics"] = diag

    # Ultrasound terms are tested only as blocks (R3-6): a single coefficient of
    # a spline or quadratic has no physiological reading.
    summary["block_tests"] = {
        "DE_vs_covariates": block_test(m_de, m_cov),
        "TFDI_vs_covariates": block_test(m_tf, m_cov),
        "DE_given_TFDI": block_test(m_joint, m_tf),
        "TFDI_given_DE": block_test(m_joint, m_de),
    }
    hc3 = {}
    for lab, m in [("DE", m_de), ("TFDI", m_tf)]:
        r = m.model.fit(cov_type="HC3")
        idx = [m.params.index.get_loc(n) for n in m.params.index if lab in n]
        hc3[lab] = float(r.wald_test(np.eye(len(m.params))[idx], scalar=True).pvalue)
    summary["block_tests_HC3_p"] = hc3

    nested = pd.DataFrame([
        {"model": "Covariates only", **checks(m_cov)},
        {"model": "Covariates + DE", **checks(m_de)},
        {"model": "Covariates + TFDI", **checks(m_tf)},
        {"model": "Covariates + DE + TFDI", **checks(m_joint)},
    ])
    nested.to_csv(OUT / "nested_models.csv", index=False)

    X = sm.add_constant(cc[["DE", "TFDI"] + COVARIATES].astype(float))
    summary["vif_joint_linear"] = {n: round(variance_inflation_factor(X.values, i), 2)
                                   for i, n in enumerate(X.columns) if n != "const"}
    summary["rho_DE_TFDI"] = round(float(stats.spearmanr(cc["DE"], cc["TFDI"]).statistic), 3)

    # Back-transformed predictions (R4-3): the new Table 4.
    pred_rows = []
    for var, res in [("DE", m_de), ("TFDI", m_tf)]:
        qs = cc[var].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).values
        g = predict_grid(res, cc, var, qs)
        g.insert(0, "percentile", ["10th", "25th", "50th", "75th", "90th"])
        g.insert(0, "index", var)
        pred_rows.append(g)
    pd.concat(pred_rows).to_csv(OUT / "table4_predicted_p01.csv", index=False)
    summary["covariate_reference_values"] = base_values(cc[COVARIATES + ["DE", "TFDI"]])

    contrasts = [iqr_contrast(f_de, cc, "DE", rng), iqr_contrast(f_tf, cc, "TFDI", rng)]
    pd.DataFrame(contrasts).to_csv(OUT / "iqr_contrasts.csv", index=False)
    summary["iqr_contrasts"] = contrasts

    # Sedation (R3-2): univariable, categorical RASS, excluding deep sedation,
    # and an exploratory interaction of each index with deep sedation.
    sed = []
    for v in ["RASS", "sedative"]:
        r = smf.ols(f"sqrtP ~ {v}", data=df).fit()
        sed.append({"analysis": f"univariable {v}", "index": "", "n": int(r.nobs),
                    "beta": round(r.params[v], 4), "p": round(r.pvalues[v], 4)})
    rho = stats.spearmanr(df["P0_1"], df["RASS"])
    summary["spearman_P01_RASS"] = {"rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 3)}
    cat_cov = COV.replace("RASS", "C(RASS_cat)")
    light = cc[cc["RASS"] > -4]
    for var, term in [("DE", de_term), ("TFDI", tf_term)]:
        full = smf.ols(f"sqrtP ~ {term} + {cat_cov}", data=cc).fit()
        red = smf.ols(f"sqrtP ~ {cat_cov}", data=cc).fit()
        sed.append({"analysis": "RASS as categories", "index": var, "n": int(full.nobs),
                    "p": block_test(full, red)["p"], "iqr_difference": iqr_point(full, cc, var)})
        full = smf.ols(f"sqrtP ~ {term} + {COV}", data=light).fit()
        red = smf.ols(f"sqrtP ~ {COV}", data=light).fit()
        sed.append({"analysis": "excluding RASS -4 and -5", "index": var, "n": int(full.nobs),
                    "p": block_test(full, red)["p"], "iqr_difference": iqr_point(full, light, var)})
        main_m = smf.ols(f"sqrtP ~ {term} + deep + {COV}", data=cc).fit()
        inter = smf.ols(f"sqrtP ~ ({term}) * deep + {COV}", data=cc).fit()
        sed.append({"analysis": "interaction with RASS -4 or -5", "index": var, "n": int(inter.nobs),
                    "p": block_test(inter, main_m)["p"]})
        sed.append({"analysis": "without sedation covariates", "index": var, "n": len(cc),
                    "iqr_difference": iqr_point(
                        smf.ols(f"sqrtP ~ {term} + age + SOFA + PaCO2", data=cc).fit(), cc, var)})
    pd.DataFrame(sed).to_csv(OUT / "sedation_sensitivity.csv", index=False)
    summary["n_rass_above_minus4"] = int(len(light))

    # The TFDI of 1.8% and the PaCO2 of 66 put back (n = 140, the R1 sample).
    rec = df[["sqrtP", "DE", "TFDI_recorded", "PaCO2_recorded"] + [c for c in COVARIATES if c != "PaCO2"]]
    rec = rec.dropna().rename(columns={"TFDI_recorded": "TFDI", "PaCO2_recorded": "PaCO2"})
    other = []
    for var, f in [("DE", f_de), ("TFDI", f_tf)]:
        r = smf.ols(f, data=rec).fit()
        other.append({"analysis": "with the recorded TFDI and PaCO2 values", "index": var, "n": int(r.nobs),
                      "r2": round(r.rsquared, 4), "iqr_difference": iqr_point(r, rec, var)})
    # Lung-injury laterality as an extra covariate (n = 120, as in R1).
    li = cc.join(df["bilateral"]).dropna(subset=["bilateral"])
    for var, term in [("DE", de_term), ("TFDI", tf_term)]:
        r = smf.ols(f"sqrtP ~ {term} + {COV} + bilateral", data=li).fit()
        other.append({"analysis": "adding lung-injury laterality", "index": var, "n": int(r.nobs),
                      "r2": round(r.rsquared, 4), "iqr_difference": iqr_point(r, li, var)})
    # All three put back (n = 141).
    full = df[["sqrtP", "DE_recorded", "TFDI_recorded", "PaCO2_recorded"] + [c for c in COVARIATES if c != "PaCO2"]]
    full = full.dropna().rename(columns={"DE_recorded": "DE", "TFDI_recorded": "TFDI", "PaCO2_recorded": "PaCO2"})
    r = smf.ols(f_de, data=full).fit()
    other.append({"analysis": "with all recorded values, including the DE of 6.1 cm", "index": "DE", "n": int(r.nobs),
                  "r2": round(r.rsquared, 4), "iqr_difference": iqr_point(r, full, "DE")})
    cooks = pd.Series(r.get_influence().cooks_distance[0], index=full.index)
    put_back = df.index[df["DE_recorded"].notna() & df["DE"].isna()]
    summary["DE_max_retained"] = round(float(df["DE"].max()), 1)
    summary["cooks_d_implausible_DE"] = round(float(cooks[put_back].max()), 1)
    pd.DataFrame(other).to_csv(OUT / "other_sensitivity.csv", index=False)
    summary["set_missing"] = [{"row": row, "variable": col, "recorded": v} for (row, col), v in IMPLAUSIBLE.items()]

    # Exploratory mortality (R3-3, R4-4): continuous P0.1, adjusted, no quartiles.
    md = df[["died", "P0_1", "SOFA", "Mean_GCS", "RASS", "age"]].dropna()
    mort = []
    for lab, f in [("unadjusted", "died ~ P0_1"),
                   ("adjusted", "died ~ P0_1 + SOFA + Mean_GCS + RASS + age")]:
        r = smf.logit(f, data=md).fit(disp=0)
        ci = r.conf_int()
        for t in r.params.index[1:]:
            mort.append({"model": lab, "term": t, "n": int(r.nobs), "events": int(md["died"].sum()),
                         "OR": round(float(np.exp(r.params[t])), 2),
                         "ci_lower": round(float(np.exp(ci.loc[t, 0])), 2),
                         "ci_upper": round(float(np.exp(ci.loc[t, 1])), 2),
                         "p_value": round(float(r.pvalues[t]), 4)})
    pd.DataFrame(mort).to_csv(OUT / "mortality_exploratory.csv", index=False)
    summary["deaths"] = int(md["died"].sum())

    # Univariable Spearman associations: descriptive only (Supplementary).
    sp = []
    for v in ["DE", "TFDI", "PaCO2", "age", "SOFA", "RASS", "Mean_GCS", "RR", "PEEP",
              "Ventilation_Duration"]:
        d = df[["P0_1", v]].dropna()
        r = stats.spearmanr(d["P0_1"], d[v])
        sp.append({"variable": v, "n": len(d), "rho": round(float(r.statistic), 3),
                   "p_value": round(float(r.pvalue), 3)})
    pd.DataFrame(sp).to_csv(OUT / "spearman_univariable.csv", index=False)

    descriptive(df, summary)
    figure2(cc, m_de, m_tf)
    supplementary_figures(df, m_de, m_tf, summary)

    with open(OUT / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return 0


def descriptive(df: pd.DataFrame, summary: dict) -> None:
    """Whole-cohort Tables 1 and 2 with descriptive P0.1 bands (no p-values).
    Oxygenation and VEP are left out until their source values are confirmed."""
    df["band"] = pd.cut(df["P0_1"], [-1, 0.999, 3.5, 99], labels=["<1.0", "1.0-3.5", ">3.5"])
    summary["band_counts"] = {k: int(v) for k, v in
                              df["band"].value_counts().reindex(["<1.0", "1.0-3.5", ">3.5"]).items()}
    summary["n_p01_ge_3_5"] = int((df["P0_1"] >= 3.5).sum())
    summary["n_peep_5"] = int((df["PEEP"] == 5).sum())
    summary["n_fio2_100"] = int((df["FiO2_pct"] == 100).sum())
    pf_kpa = df["PF_arterial"].dropna() * 7.50062
    summary["oxygenation_unreported"] = {
        "n_pao2_values_differ": int(df["PaO2"].ne(df["PaO2.1"]).sum()),
        "PaO2_arterial_median": round(float(df["PaO2_arterial"].median()), 1),
        "PaO2_arterial_max": round(float(df["PaO2_arterial"].max()), 1),
        "PF_arterial_median": round(float(df["PF_arterial"].median()), 1),
        "PF_arterial_max": round(float(df["PF_arterial"].max()), 1),
        "n_pf_over_300_if_kpa": int((pf_kpa > 300).sum()),
        "n_pf": len(pf_kpa),
    }

    def med(s):
        s = s.dropna()
        return f"{s.median():.2f} [{s.quantile(.25):.2f}–{s.quantile(.75):.2f}]"

    def cnt(s, level):
        s = s.dropna()
        k = int((s == level).sum())
        return f"{k} ({100 * k / len(s):.1f})"

    groups = [("Overall", df)] + [(b, df[df["band"] == b]) for b in ["<1.0", "1.0-3.5", ">3.5"]]
    t1_cont = {"age": "Age, years", "BMI": "BMI, kg/m²", "SOFA": "SOFA score",
               "Mean_GCS": "Glasgow Coma Scale", "RASS": "RASS score",
               "Ventilation_Duration": "Duration of mechanical ventilation, days"}
    t1_cat = [("gender", "male", "Male sex"), ("cigarette", "yes", "Smoking history"),
              ("substance", "yes", "Substance use"), ("comorbitity", "yes", "Any comorbidity"),
              ("Admission_Diagnosis", "pulmonary", "Admission diagnosis: pulmonary"),
              ("Admission_Diagnosis", "surgery", "Admission diagnosis: surgical"),
              ("Admission_Diagnosis", "cardiovascular", "Admission diagnosis: cardiovascular"),
              ("Admission_Diagnosis", "other", "Admission diagnosis: other"),
              ("Lung_Injury_Status", "bilateral", "Bilateral lung injury"),
              ("midazolam", "yes", "Midazolam"), ("fendanyl", "yes", "Fentanyl"),
              ("morphine", "yes", "Morphine"), ("methadone", "yes", "Methadone"),
              ("sedative", 1, "Any sedative or opioid"), ("deep", 1, "RASS −4 or −5"),
              ("died", 1, "In-hospital death")]
    t2_cont = {"P0_1": "P0.1, cmH₂O", "DE": "Diaphragmatic excursion, cm",
               "TFDI": "Diaphragm thickening fraction, %", "RR": "Total respiratory rate, breaths/min",
               "PEEP": "PEEP, cmH₂O", "VR": "Ventilatory ratio", "PH": "pH",
               "PaCO2": "PaCO₂, mmHg", "HCO3": "HCO₃⁻, mmol/L"}
    for name, cont, cat in [("table1_whole_cohort.csv", t1_cont, t1_cat),
                            ("table2_whole_cohort.csv", t2_cont, [])]:
        rows = [{"variable": "n", **{g: str(len(d)) for g, d in groups}}]
        for col, lab in cont.items():
            rows.append({"variable": lab, **{g: med(d[col]) for g, d in groups}})
        for col, lvl, lab in cat:
            rows.append({"variable": lab, **{g: cnt(d[col], lvl) for g, d in groups}})
        pd.DataFrame(rows).to_csv(OUT / name, index=False)


def figure2(cc: pd.DataFrame, m_de, m_tf) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, var, res, xl, tag in [
            (axes[0], "TFDI", m_tf, "Diaphragm thickening fraction, TFDI (%)", "a"),
            (axes[1], "DE", m_de, "Diaphragmatic excursion, DE (cm)", "b")]:
        ax.scatter(cc[var], cc["P0_1"], s=14, alpha=0.6, color="#3b6ea5", edgecolor="none")
        xs = np.linspace(cc[var].quantile(0.02), cc[var].quantile(0.98), 80)
        g = predict_grid(res, cc, var, xs)
        ax.plot(xs, g["pred_P01"], color="#b03a2e", lw=2)
        ax.fill_between(xs, g["ci_lower"], g["ci_upper"], color="#b03a2e", alpha=0.18, lw=0)
        ax.set_xlabel(xl)
        ax.set_ylabel("P0.1 (cmH$_2$O)")
        ax.set_title(tag, loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure2_separate_models.png", bbox_inches="tight")
    plt.close(fig)


def supplementary_figures(df: pd.DataFrame, m_de, m_tf, summary: dict) -> None:
    """Fig S1: residual diagnostics of the two primary models. Fig S2: P0.1 by RASS."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for row, (lab, m) in enumerate([("DE model", m_de), ("TFDI model", m_tf)]):
        ax = axes[row, 0]
        ax.scatter(m.fittedvalues, m.resid, s=12, alpha=0.6, color="#3b6ea5", edgecolor="none")
        ax.axhline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel("Fitted \u221aP0.1")
        ax.set_ylabel("Residual")
        ax.set_title(f"{'ac'[row]}  {lab}: residuals vs fitted", loc="left", fontsize=10)
        sm.qqplot(m.resid, line="s", ax=axes[row, 1], markersize=3, alpha=0.6)
        axes[row, 1].set_title(f"{'bd'[row]}  {lab}: normal Q-Q", loc="left", fontsize=10)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figureS1_residuals.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    d = df[["P0_1", "RASS"]].dropna()
    levels = sorted(d["RASS"].unique())
    kw = stats.kruskal(*[d.loc[d["RASS"] == v, "P0_1"] for v in levels])
    summary["kruskal_P01_RASS"] = {"H": round(float(kw.statistic), 2), "p": round(float(kw.pvalue), 3),
                                   "levels": [int(v) for v in levels]}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot([d.loc[d["RASS"] == v, "P0_1"] for v in levels], positions=levels,
               widths=0.6, showfliers=False)
    jitter = np.random.default_rng(SEED).uniform(-0.15, 0.15, len(d))
    ax.scatter(d["RASS"] + jitter, d["P0_1"], s=10, alpha=0.5, color="#3b6ea5", edgecolor="none")
    ax.set_xlabel("RASS score")
    ax.set_ylabel("P0.1 (cmH$_2$O)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figureS2_p01_by_rass.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        sys.exit(main())
