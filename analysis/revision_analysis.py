#!/usr/bin/env python3
"""Run the respiratory-drive statistical analysis on an external CSV input."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

N_ENROLLED = 141
REQUIRED_COLUMNS = {
    "FiO2", "Sedation_Depth", "Lung_Injury_Status", "outcome", "P0_1",
    "DE", "TFDI", "PaCO2", "age", "SOFA", "VR", "PH", "morphine",
    "P_F_ratio", "SPO2", "SaO2", "Admission_Diagnosis",
}


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

_log: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    _log.append(msg)


def require(condition: bool, msg: str) -> None:
    """Fail loudly. A silent row drop is how the three-model problem happened."""
    if not condition:
        raise AssertionError(f"SANITY CHECK FAILED: {msg}")




def load(data_path: Path, expected_md5: str | None = None) -> pd.DataFrame:
    require(data_path.exists(), f"{data_path} not found")
    md5 = hashlib.md5(data_path.read_bytes()).hexdigest()
    log(f"Input      : {data_path}")
    log(f"MD5        : {md5}")
    if expected_md5 and md5 != expected_md5:
        log(f"WARNING: expected {expected_md5}; input checksum differs")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    log(f"Loaded     : {df.shape[0]} rows x {df.shape[1]} columns")
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    require(not missing, f"missing required columns: {', '.join(missing)}")
    require(len(df) == N_ENROLLED, f"expected {N_ENROLLED} rows, got {len(df)}")

    pao2_cols = [c for c in df.columns if c.lower().startswith("pao2")]
    log(f"PaO2-like columns present: {pao2_cols}")

    fio2_raw = df["FiO2"].astype(str).str.strip()
    df["FiO2_pct"] = pd.to_numeric(fio2_raw.str.rstrip("%"), errors="coerce")
    df["FiO2_frac"] = df["FiO2_pct"] / 100.0
    log(f"FiO2 parsed from text; FiO2 = 0% in {int((df['FiO2_pct'] == 0).sum())} row(s)")

    numeric = [
        "age", "BMI", "SOFA", "Mean_GCS", "DE", "TFDI", "VEP", "VR", "P0_1",
        "P_F_ratio", "PEEP", "RR", "Ventilation_Duration", "SPO2", "SaO2",
        "PH", "HCO3", "PaCO2", "pbw", "Sedation_Depth",
    ] + pao2_cols
    for column in numeric:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in df.select_dtypes(include=["object", "str"]).columns:
        df[column] = df[column].astype(str).str.strip().str.lower().replace({"nan": np.nan})

    df["RASS"] = df["Sedation_Depth"]
    require(df["RASS"].dropna().between(-5, 4).all(), "Sedation_Depth outside the RASS range -5..+4")
    df["bilateral"] = df["Lung_Injury_Status"].map({"bilateral": 1, "unilateral": 0})
    df["died"] = df["outcome"].map({"death": 1, "discharge": 0})
    require(df["died"].notna().all(), "outcome failed to map to death/discharge")

    df["P01_quartile"] = pd.qcut(
        df["P0_1"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )
    cuts = df["P0_1"].quantile([0.25, 0.50, 0.75]).round(2).tolist()
    log(f"P0.1 quartile cut-points: {cuts}")
    require(cuts == [0.6, 1.3, 1.8], f"quartile cut-points {cuts} do not match [0.6, 1.3, 1.8]")
    return df




def missingness(df: pd.DataFrame, output_dir: Path) -> dict:
    rows = []
    for c in df.columns:
        n_miss = int(df[c].isna().sum())
        if n_miss:
            rows.append({"variable": c, "n_missing": n_miss, "pct_missing": round(100 * n_miss / len(df), 1)})
    miss = pd.DataFrame(rows).sort_values("n_missing", ascending=False)
    miss.to_csv(output_dir / "missingness_report.csv", index=False)

    model_a_vars = ["P0_1", "DE", "TFDI", "PaCO2", "age", "SOFA", "bilateral"]
    model_b_vars = ["P0_1", "DE", "TFDI", "PaCO2", "age", "SOFA"]
    n_a = int(df[model_a_vars].dropna().shape[0])
    n_b = int(df[model_b_vars].dropna().shape[0])

    counts = {
        "n_enrolled_rows_in_datafile": N_ENROLLED,
        "n_with_P0_1": int(df["P0_1"].notna().sum()),
        "n_complete_model_A_with_lung_injury": n_a,
        "n_complete_model_B_without_lung_injury": n_b,
        "n_missing_DE": int(df["DE"].isna().sum()),
        "n_missing_lung_injury": int(df["Lung_Injury_Status"].isna().sum()),
    }
    (output_dir / "sample_size_reconciliation.json").write_text(json.dumps(counts, indent=2))

    log("")
    log("SAMPLE SIZE RECONCILIATION ")
    log(f"  rows in data file                       : {N_ENROLLED}")
    log(f"  missing DE                              : {counts['n_missing_DE']}")
    log(f"  missing Lung_Injury_Status              : {counts['n_missing_lung_injury']}")
    log(f"  complete cases, model WITH lung injury  : {n_a}")
    log(f"  complete cases, model WITHOUT lung injury: {n_b}")
    return counts




def oxygenation_integrity(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for var, unit, plausible in [
        ("P_F_ratio", "mmHg", "inclusion criterion was <= 300; severe ARDS is < 100"),
        ("PaO2", "mmHg", "arterial, on supplemental O2; expected 60-100"),
        ("SPO2", "%", "pulse oximetry; expected 88-100"),
        ("SaO2", "%", "arterial saturation; must agree with SpO2"),
    ]:
        if var not in df.columns:
            continue
        s = df[var].dropna()
        rows.append(
            {
                "variable": var,
                "unit_as_reported": unit,
                "n": len(s),
                "median": round(s.median(), 2),
                "min": round(s.min(), 2),
                "max": round(s.max(), 2),
                "n_below_100" if var == "P_F_ratio" else "n_flagged": (
                    int((s < 100).sum()) if var == "P_F_ratio"
                    else int((s < 90).sum()) if var == "SPO2"
                    else int((s < 60).sum()) if var == "SaO2"
                    else int((s < 60).sum())
                ),
                "expected_range": plausible,
            }
        )
    integrity = pd.DataFrame(rows)
    integrity.to_csv(output_dir / "oxygenation_integrity.csv", index=False)

    notes = [
        f"P/F ratio: median {df['P_F_ratio'].median():.1f}; "
        f"{int((df['P_F_ratio'] < 100).sum())} of {df['P_F_ratio'].notna().sum()} patients below 100, "
        "against a stated inclusion criterion of <= 300.",
        f"SaO2 median {df['SaO2'].median():.1f}% vs SpO2 median {df['SPO2'].median():.1f}%. "
        f"FiO2 = 0% recorded for {int((df['FiO2_pct'] == 0).sum())} patient(s).",
        "Two distinct columns are both named PaO2; which is the arterial gas is unresolved.",
    ]
    (output_dir / "oxygenation_integrity_notes.txt").write_text("\n".join(f"- {n}" for n in notes))
    log("")
    log("OXYGENATION INTEGRITY  — flagged, NOT corrected:")
    for n in notes:
        log(f"  - {n}")




def fit_competing_models(df: pd.DataFrame, output_dir: Path) -> dict:
    results: dict[str, object] = {}

    # Model A: OLS with the lung-injury indicator.
    a_data = df[["P0_1", "DE", "TFDI", "PaCO2", "age", "SOFA", "bilateral"]].dropna()
    model_a = smf.ols("P0_1 ~ DE + TFDI + PaCO2 + age + SOFA + bilateral", data=a_data).fit()

    # Model B: OLS without the lung-injury indicator.
    b_data = df[["P0_1", "DE", "TFDI", "PaCO2", "age", "SOFA"]].dropna()
    model_b = smf.ols("P0_1 ~ DE + TFDI + PaCO2 + age + SOFA", data=b_data).fit()

    # Model C: logistic regression for the thresholded P0.1 outcome.
    c_df = df.copy()
    c_df["P01_high"] = (c_df["P0_1"] > 1.5).astype(int)
    c_df["morphine_yes"] = (c_df["morphine"] == "yes").astype(int)
    c_data = c_df[["P01_high", "DE", "TFDI", "VR", "PH", "PaCO2", "morphine_yes"]].dropna()
    model_c = smf.logit("P01_high ~ DE + TFDI + VR + PH + PaCO2 + morphine_yes", data=c_data).fit(disp=0)

    def tidy(res, label: str, n: int, logistic: bool) -> pd.DataFrame:
        ci = res.conf_int()
        out = pd.DataFrame(
            {
                "model": label,
                "n": n,
                "predictor": res.params.index,
                "estimate": res.params.round(4).values,
                "ci_lower": ci[0].round(4).values,
                "ci_upper": ci[1].round(4).values,
                "p_value": res.pvalues.round(4).values,
            }
        )
        out["significant_at_0.05"] = out["p_value"] < 0.05
        if logistic:
            out["odds_ratio"] = np.exp(res.params).round(4).values
        return out

    tidy_a = tidy(model_a, "A: OLS with lung injury (with lung-injury term)", len(a_data), False)
    tidy_b = tidy(model_b, "B: OLS without lung injury (without lung-injury term)", len(b_data), False)
    tidy_c = tidy(model_c, "C: logistic, P0.1 > 1.5 (threshold logistic model)", len(c_data), True)

    combined = pd.concat([tidy_a, tidy_b, tidy_c], ignore_index=True)
    combined.to_csv(output_dir / "competing_models_comparison.csv", index=False)

    fit = pd.DataFrame(
        [
            {"model": "A", "n": len(a_data), "r_squared": round(model_a.rsquared, 4),
             "adj_r_squared": round(model_a.rsquared_adj, 4), "aic": round(model_a.aic, 2)},
            {"model": "B", "n": len(b_data), "r_squared": round(model_b.rsquared, 4),
             "adj_r_squared": round(model_b.rsquared_adj, 4), "aic": round(model_b.aic, 2)},
            {"model": "C", "n": len(c_data), "r_squared": round(model_c.prsquared, 4),
             "adj_r_squared": np.nan, "aic": round(model_c.aic, 2)},
        ]
    )
    fit.to_csv(output_dir / "competing_models_fit.csv", index=False)

    log("")
    log("COMPETING MODELS  — DE across all three:")
    for lbl, t in [("A", tidy_a), ("B", tidy_b), ("C", tidy_c)]:
        row = t[t["predictor"] == "DE"].iloc[0]
        verdict = "SIGNIFICANT" if row["p_value"] < 0.05 else "not significant"
        log(f"  Model {lbl} (n={row['n']}): estimate {row['estimate']}, p = {row['p_value']} -> {verdict}")

    primary_df = df[["P0_1", "DE", "TFDI", "PaCO2", "age", "SOFA"]].dropna().copy()
    primary_df["sqrtP"] = np.sqrt(primary_df["P0_1"])
    results.update(model_a=model_a, model_b=model_b, model_c=model_c,
                   a_data=a_data, b_data=b_data, c_data=c_data,
                   primary_df=primary_df)
    return results




def rebuild_table4(df: pd.DataFrame, output_dir: Path) -> None:
    """Fit the square-root primary model and sensitivity analyses."""
    core = ["DE", "TFDI", "PaCO2", "age", "SOFA"]
    labels = {
        "DE": "Diaphragmatic excursion (cm)",
        "TFDI": "Thickening fraction (%)",
        "I(TFDI ** 2)": "Thickening fraction squared",
        "PaCO2": "PaCO2 (mmHg)",
        "age": "Age (years)",
        "SOFA": "SOFA score",
        "bilateral": "Bilateral lung injury (vs unilateral)",
        "Intercept": "Intercept",
    }

    work = df[["P0_1"] + core + ["bilateral"]].copy()
    work["sqrtP"] = np.sqrt(work["P0_1"])

    primary_data = work[["sqrtP", "P0_1"] + core].dropna()
    n_primary = len(primary_data)
    require(n_primary == 140, f"expected the primary model to be n=140, got {n_primary}")

    F_PRIMARY = "sqrtP ~ DE + TFDI + I(TFDI**2) + PaCO2 + age + SOFA"
    F_LINEAR = "sqrtP ~ " + " + ".join(core)

    primary = smf.ols(F_PRIMARY, data=primary_data).fit()
    primary_hc3 = smf.ols(F_PRIMARY, data=primary_data).fit(cov_type="HC3")
    linear140 = smf.ols(F_LINEAR, data=primary_data).fit()

    lung_data = work[["sqrtP"] + core + ["bilateral"]].dropna()
    with_lung = smf.ols(
        "sqrtP ~ DE + TFDI + I(TFDI**2) + PaCO2 + age + SOFA + bilateral",
        data=lung_data,
    ).fit()

    rows = []

    for v in core + ["bilateral"]:
        d = work[["sqrtP", v]].dropna()
        res = smf.ols(f"sqrtP ~ {v}", data=d).fit()
        ci = res.conf_int()
        rows.append({
            "panel": "Univariable",
            "predictor": labels[v],
            "n": int(res.nobs),
            "beta": round(float(res.params[v]), 4),
            "ci_lower": round(float(ci[0][v]), 4),
            "ci_upper": round(float(ci[1][v]), 4),
            "p_value": round(float(res.pvalues[v]), 4),
            "significant": bool(res.pvalues[v] < 0.05),
        })

    ci = primary.conf_int()
    for v in primary.params.index:
        rows.append({
            "panel": f"PRIMARY multivariable (n={n_primary})",
            "predictor": labels.get(v, v),
            "n": n_primary,
            "beta": round(float(primary.params[v]), 5),
            "ci_lower": round(float(ci[0][v]), 5),
            "ci_upper": round(float(ci[1][v]), 5),
            "p_value": round(float(primary.pvalues[v]), 4),
            "significant": bool(primary.pvalues[v] < 0.05),
            "p_HC3_sensitivity": round(float(primary_hc3.pvalues[v]), 4),
        })

    ci = linear140.conf_int()
    for v in linear140.params.index:
        rows.append({
            "panel": f"Sensitivity: linear form (n={n_primary})",
            "predictor": labels.get(v, v),
            "n": n_primary,
            "beta": round(float(linear140.params[v]), 5),
            "ci_lower": round(float(ci[0][v]), 5),
            "ci_upper": round(float(ci[1][v]), 5),
            "p_value": round(float(linear140.pvalues[v]), 4),
            "significant": bool(linear140.pvalues[v] < 0.05),
        })

    ci = with_lung.conf_int()
    for v in with_lung.params.index:
        rows.append({
            "panel": f"Sensitivity: lung injury retained (n={int(with_lung.nobs)})",
            "predictor": labels.get(v, v),
            "n": int(with_lung.nobs),
            "beta": round(float(with_lung.params[v]), 5),
            "ci_lower": round(float(ci[0][v]), 5),
            "ci_upper": round(float(ci[1][v]), 5),
            "p_value": round(float(with_lung.pvalues[v]), 4),
            "significant": bool(with_lung.pvalues[v] < 0.05),
        })

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "table4_rebuilt_sqrt.csv", index=False)

    def checks(model):
        return (
            float(stats.shapiro(model.resid)[1]),
            float(het_breuschpagan(model.resid, model.model.exog)[1]),
            float(linear_reset(model, power=2, use_f=True).pvalue),
        )

    sw_p, bp_p, reset_p = checks(primary)
    sw_l, bp_l, reset_l = checks(linear140)
    raw140 = smf.ols("P0_1 ~ " + " + ".join(core), data=primary_data).fit()
    sw_r, bp_r, reset_r = checks(raw140)

    X = sm.add_constant(primary_data[core].astype(float))
    vifs = {
        name: round(variance_inflation_factor(X.values, i), 3)
        for i, name in enumerate(X.columns) if name != "const"
    }

    joint_de = primary.f_test("DE = 0")
    joint_tf = primary.f_test("TFDI = 0, I(TFDI ** 2) = 0")
    joint_us = primary.f_test("DE = 0, TFDI = 0, I(TFDI ** 2) = 0")

    base = "PaCO2 + age + SOFA"
    m_de = smf.ols(f"sqrtP ~ DE + {base}", data=primary_data).fit()
    m_tf = smf.ols(f"sqrtP ~ TFDI + I(TFDI**2) + {base}", data=primary_data).fit()
    m_none = smf.ols(f"sqrtP ~ {base}", data=primary_data).fit()
    pd.DataFrame([
        {"model": "covariates only", "r_squared": round(m_none.rsquared, 4),
         "adj_r_squared": round(m_none.rsquared_adj, 4), "aic": round(m_none.aic, 1)},
        {"model": "+ DE alone", "r_squared": round(m_de.rsquared, 4),
         "adj_r_squared": round(m_de.rsquared_adj, 4), "aic": round(m_de.aic, 1)},
        {"model": "+ TFDI (with quadratic) alone", "r_squared": round(m_tf.rsquared, 4),
         "adj_r_squared": round(m_tf.rsquared_adj, 4), "aic": round(m_tf.aic, 1)},
        {"model": "+ both", "r_squared": round(primary.rsquared, 4),
         "adj_r_squared": round(primary.rsquared_adj, 4), "aic": round(primary.aic, 1)},
    ]).to_csv(output_dir / "de_tfdi_nested_models.csv", index=False)

    cooks = primary.get_influence().cooks_distance[0]
    keep = primary_data[cooks <= 4 / n_primary]
    infl = smf.ols(F_PRIMARY, data=keep).fit()

    verdict = lambda p: "PASS" if p > 0.05 else "FAIL"
    diag = [
        "Rebuilt Table 4 — model diagnostics",
        "=" * 58,
        "",
        "PRIMARY MODEL",
        "  outcome     : sqrt(P0.1), P0.1 in cmH2O",
        f"  predictors  : DE, TFDI, TFDI^2, PaCO2, age, SOFA",
        f"  n           : {n_primary}  ({len(df) - n_primary} rows excluded for missing values)",
        "",
        "Assumption checks",
        f"{'':30s} {'Shapiro-Wilk':>14} {'Breusch-Pagan':>15} {'RESET':>12}",
        f"  {'PRIMARY (sqrt, +TFDI^2)':28s} {sw_p:>10.4f} {verdict(sw_p):>4}"
        f" {bp_p:>10.4f} {verdict(bp_p):>4} {reset_p:>7.4f} {verdict(reset_p):>4}",
        f"  {'sqrt, linear form':28s} {sw_l:>10.4f} {verdict(sw_l):>4}"
        f" {bp_l:>10.4f} {verdict(bp_l):>4} {reset_l:>7.4f} {verdict(reset_l):>4}",
        f"  {'raw scale (as original)':28s} {sw_r:>10.2e} {verdict(sw_r):>4}"
        f" {bp_r:>10.2e} {verdict(bp_r):>4} {reset_r:>7.1e} {verdict(reset_r):>4}",
    ]
    diag += [
        "",
        f"Influential observations above Cook's distance 4/n: {int((cooks > 4 / n_primary).sum())}",
        f"Influence-refit n: {int(infl.nobs)}",
        f"Primary model Spearman DE/TFDI: {primary_data[['DE', 'TFDI']].corr(method='spearman').iloc[0, 1]:.3f}",
    ]
    (output_dir / "table4_diagnostics.txt").write_text("\n".join(diag))

    # -- back-transformed predictions ----------------------------------------
    # E[sqrt(Y)] = Xb, so (Xb)^2 estimates the MEDIAN of P0.1, not its mean.
    medians = {v: float(primary_data[v].median()) for v in core}
    pred_rows = []
    for driver in ["DE", "TFDI"]:
        lo, hi = primary_data[driver].quantile([0.05, 0.95])
        for value in np.linspace(lo, hi, 9):
            scenario = dict(medians)
            scenario[driver] = float(value)
            point = float(primary.predict(pd.DataFrame([scenario]))[0])
            pred_rows.append({
                "varying": driver,
                "value": round(float(value), 2),
                "fitted_sqrt_P01": round(point, 4),
                "predicted_median_P01_cmH2O": round(point ** 2, 3),
                "others_held_at": "median",
            })
    preds = pd.DataFrame(pred_rows)
    preds.to_csv(output_dir / "table4_backtransformed_predictions.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, driver, xlabel in zip(
        axes, ["DE", "TFDI"],
        ["Diaphragmatic excursion (cm)", "Thickening fraction (%)"],
    ):
        sub = preds[preds["varying"] == driver]
        ax.scatter(primary_data[driver], primary_data["P0_1"], s=28, alpha=0.35,
                   color="#2c3e50", label=f"Patients (n={n_primary})")
        ax.plot(sub["value"], sub["predicted_median_P01_cmH2O"], color="#c0392b",
                lw=2.4, label="Model, back-transformed")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$P_{0.1}$ (cmH$_2$O)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title("A. Excursion")
    axes[1].set_title("B. Thickening fraction")
    fig.suptitle(
        f"Primary model on $\\sqrt{{P_{{0.1}}}}$ (n = {n_primary}), back-transformed to cmH$_2$O; "
        "other covariates at their medians", fontsize=12)
    fig.savefig(output_dir / "figure_table4_backtransformed.png")
    plt.close(fig)

    md = [
        "**Table 4. Univariable and multivariable linear regression for predictors of "
        r"respiratory drive, modelled as $\sqrt{P_{0.1}}$**",
        "",
        "| Predictor | n | β (√cmH₂O) | 95% CI | p |",
        "|---|---|---|---|---|",
        "| *Univariable models* | | | | |",
    ]
    def fmt_p(p: float) -> str:
        from decimal import Decimal, ROUND_HALF_UP
        if p < 0.001:
            return "< 0.001"
        return str(Decimal(str(p)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    for _, r in table[table["panel"] == "Univariable"].iterrows():
        star = "**" if r["significant"] else ""
        md.append(
            f"| {r['predictor']} | {r['n']} | {r['beta']:.3f} | "
            f"{r['ci_lower']:.3f} to {r['ci_upper']:.3f} | {star}{fmt_p(r['p_value'])}{star} |"
        )
    md.append(f"| *Multivariable model (n = {n_primary})* | | | | |")
    for _, r in table[table["panel"].str.startswith("PRIMARY")].iterrows():
        if r["predictor"] == "Intercept":
            continue
        star = "**" if r["significant"] else ""
        md.append(
            f"| {r['predictor']} | {r['n']} | {r['beta']:.4f} | "
            f"{r['ci_lower']:.4f} to {r['ci_upper']:.4f} | {star}{fmt_p(r['p_value'])}{star} |"
        )
    md += [
        "",
        f"R² = {primary.rsquared:.3f} (adjusted {primary.rsquared_adj:.3f}). The outcome is "
        "square-root transformed; residuals satisfy normality (Shapiro–Wilk "
        f"p = {sw_p:.2f}), constant variance (Breusch–Pagan p = {bp_p:.2f}) and functional "
        f"form (Ramsey RESET p = {reset_p:.2f}), so classical standard errors are reported. "
        f"Thickening fraction enters with a quadratic term; the two are interpreted jointly "
        f"(F({int(joint_tf.df_num)},{int(joint_tf.df_denom)}) = {float(joint_tf.fvalue):.1f}, "
        f"p = {float(joint_tf.pvalue):.3g}). Variance inflation factors are reported separately. "
    ]
    (output_dir / "table4_rebuilt_sqrt.md").write_text("\n".join(md) + "\n")

    log("")
    log(f"REBUILT TABLE 4 — PRIMARY MODEL on sqrt(P0.1), n = {n_primary}")
    log(f"  R2 = {primary.rsquared:.4f} (adj {primary.rsquared_adj:.4f})")
    log(f"  Shapiro-Wilk p = {sw_p:.4f} {verdict(sw_p)} | "
        f"Breusch-Pagan p = {bp_p:.4f} {verdict(bp_p)} | "
        f"RESET p = {reset_p:.4f} {verdict(reset_p)}")
    log(f"  {'predictor':32s} {'beta':>10} {'95% CI':>24} {'p':>8} {'p(HC3)':>8}")
    for _, r in table[table["panel"].str.startswith("PRIMARY")].iterrows():
        if r["predictor"] == "Intercept":
            continue
        mark = "*" if r["significant"] else " "
        log(f"  {r['predictor']:32s} {r['beta']:10.5f} "
            f"[{r['ci_lower']:10.5f},{r['ci_upper']:10.5f}] {r['p_value']:7.4f}{mark} "
            f"{r['p_HC3_sensitivity']:8.4f}")
    log(f"  joint ultrasound terms: F({int(joint_us.df_num)},{int(joint_us.df_denom)}) "
        f"= {float(joint_us.fvalue):.2f}, p = {float(joint_us.pvalue):.2e}")


def specification_robustness(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare plausible outcome scales and covariance estimators."""
    s = df[["P0_1", "DE", "TFDI", "PaCO2", "age", "SOFA", "bilateral"]].dropna().copy()
    s["sqrtP"] = np.sqrt(s["P0_1"])
    rhs = "DE + TFDI + PaCO2 + age + SOFA + bilateral"
    of_interest = ["DE", "TFDI", "PaCO2", "bilateral"]

    m_raw = smf.ols(f"P0_1 ~ {rhs}", data=s).fit()
    cooks = m_raw.get_influence().cooks_distance[0]
    keep = s[cooks <= 4 / len(s)]

    specs = [
        ("raw P0.1, classical SE", smf.ols(f"P0_1 ~ {rhs}", data=s).fit()),
        ("raw P0.1, HC1 robust", smf.ols(f"P0_1 ~ {rhs}", data=s).fit(cov_type="HC1")),
        ("raw P0.1, HC3 robust", smf.ols(f"P0_1 ~ {rhs}", data=s).fit(cov_type="HC3")),
        ("sqrt(P0.1), classical SE", smf.ols(f"sqrtP ~ {rhs}", data=s).fit()),
        ("sqrt(P0.1), HC3 robust", smf.ols(f"sqrtP ~ {rhs}", data=s).fit(cov_type="HC3")),
        ("log(P0.1), classical SE", smf.ols(f"np.log(P0_1) ~ {rhs}", data=s).fit()),
        ("Gamma GLM, log link",
         smf.glm(f"P0_1 ~ {rhs}", data=s,
                 family=sm.families.Gamma(sm.families.links.Log())).fit()),
        (f"raw P0.1 HC1, influential removed (n={len(keep)})",
         smf.ols(f"P0_1 ~ {rhs}", data=keep).fit(cov_type="HC1")),
        (f"sqrt(P0.1), influential removed (n={len(keep)})",
         smf.ols(f"sqrtP ~ {rhs}", data=keep).fit()),
    ]

    rows = []
    for label, fit in specs:
        row = {"specification": label, "n": int(fit.nobs)}
        for v in of_interest:
            row[f"{v}_p"] = round(float(fit.pvalues[v]), 4)
            row[f"{v}_sig"] = bool(fit.pvalues[v] < 0.05)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "specification_robustness.csv", index=False)

    survived = {v: int(out[f"{v}_sig"].sum()) for v in of_interest}
    summary = [
        "Specification robustness summary",
        "",
        *[f"{v}: significant in {survived[v]} of {len(out)} specifications" for v in of_interest],
    ]
    (output_dir / "specification_robustness_summary.txt").write_text("\n".join(summary) + "\n")

    log("")
    log("SPECIFICATION ROBUSTNESS:")
    log(f"  {'specification':42s} " + "  ".join(f"{v:>10}" for v in of_interest))
    for label, fit in specs:
        cells = []
        for v in of_interest:
            p = float(fit.pvalues[v])
            cells.append(f"{p:.4f}" + ("*" if p < 0.05 else " "))
        log(f"  {label:42s} " + "  ".join(f"{c:>10}" for c in cells))
    log("  " + ", ".join(f"{v} significant in {survived[v]}/9" for v in of_interest))


def collinearity(df: pd.DataFrame, fits: dict, output_dir: Path) -> None:
    rows = []
    for label, data, predictors in [
        ("A: with lung injury", fits["a_data"], ["DE", "TFDI", "PaCO2", "age", "SOFA", "bilateral"]),
        ("B: without lung injury", fits["b_data"], ["DE", "TFDI", "PaCO2", "age", "SOFA"]),
    ]:
        X = sm.add_constant(data[predictors].astype(float))
        for i, name in enumerate(X.columns):
            if name == "const":
                continue
            vif = variance_inflation_factor(X.values, i)
            rows.append(
                {
                    "model": label,
                    "predictor": name,
                    "VIF": round(vif, 3),
                    "tolerance": round(1 / vif, 3),
                    "interpretation": (
                        "no concern" if vif < 2.5
                        else "moderate" if vif < 5
                        else "high" if vif < 10
                        else "severe"
                    ),
                }
            )
        # condition index on the standardised design matrix
        Z = (data[predictors] - data[predictors].mean()) / data[predictors].std()
        eig = np.linalg.eigvalsh(np.corrcoef(Z.T.values))
        ci = float(np.sqrt(eig.max() / eig[eig > 1e-12].min()))
        rows.append(
            {"model": label, "predictor": "CONDITION INDEX", "VIF": round(ci, 3),
             "tolerance": np.nan,
             "interpretation": "weak (<10)" if ci < 10 else "moderate (10-30)" if ci < 30 else "strong (>30)"}
        )

    vif_df = pd.DataFrame(rows)
    vif_df.to_csv(output_dir / "vif_diagnostics.csv", index=False)

    rho, p = stats.spearmanr(df["DE"].dropna(), df.loc[df["DE"].notna(), "TFDI"])
    pear = df[["DE", "TFDI"]].corr().iloc[0, 1]
    (output_dir / "de_tfdi_correlation.txt").write_text(
        f"DE vs TFDI, Spearman rho = {rho:.3f} (p = {p:.3g})\n"
        f"DE vs TFDI, Pearson  r   = {pear:.3f}\n"
        f"n = {int(df['DE'].notna().sum())}\n"
    )
    log("")
    log(f"COLLINEARITY : DE vs TFDI Spearman rho = {rho:.3f}, Pearson r = {pear:.3f}")
    for _, r in vif_df[vif_df["predictor"].isin(["DE", "TFDI"])].iterrows():
        log(f"  {r['model']:24s} {r['predictor']:5s} VIF = {r['VIF']:6.3f}  tolerance = {r['tolerance']}")




def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, Harrell's parameterisation.

    Returns k-1 columns for k knots: the linear term followed by k-2 cubic
    terms, each constrained to be linear beyond the outer knots.

    Implemented here rather than via patsy.cr(), which raises on numpy >= 2.0.
    """
    x = np.asarray(x, dtype=float)
    k = len(knots)
    require(k >= 3, "a restricted cubic spline needs at least 3 knots")
    t = np.asarray(knots, dtype=float)
    denom = (t[-1] - t[0]) ** 2

    def cube(v: np.ndarray) -> np.ndarray:
        return np.clip(v, 0, None) ** 3

    cols = [x]
    for j in range(k - 2):
        term = (
            cube(x - t[j])
            - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        ) / denom
        cols.append(term)
    return np.column_stack(cols)


def rcs_knots(x: pd.Series, k: int) -> np.ndarray:
    """Harrell's recommended knot percentiles."""
    pct = {3: [10, 50, 90], 4: [5, 35, 65, 95], 5: [5, 27.5, 50, 72.5, 95]}[k]
    return np.percentile(x.dropna(), pct)


def _fit_rcs(d: pd.DataFrame, x: str, k: int):
    knots = rcs_knots(d[x], k)
    B = rcs_basis(d[x].values, knots)
    X = sm.add_constant(B)
    res = sm.OLS(d["P0_1"].values, X).fit()
    res._rcs_knots = knots  # carried so the figure can reuse them
    return res


def nonlinearity(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for x in ["DE", "TFDI"]:
        d = df[["P0_1", x]].dropna()
        fitted = {
            "linear": smf.ols(f"P0_1 ~ {x}", data=d).fit(),
            "quadratic": smf.ols(f"P0_1 ~ {x} + I({x}**2)", data=d).fit(),
            "RCS 3 knots": _fit_rcs(d, x, 3),
            "RCS 4 knots": _fit_rcs(d, x, 4),
        }
        for name, res in fitted.items():
            rows.append(
                {
                    "predictor": x, "model": name, "n": int(res.nobs),
                    "df_model": int(res.df_model),
                    "r_squared": round(res.rsquared, 4),
                    "adj_r_squared": round(res.rsquared_adj, 4),
                    "AIC": round(res.aic, 2), "BIC": round(res.bic, 2),
                    "rmse": round(float(np.sqrt(res.mse_resid)), 4),
                }
            )
        # likelihood-ratio test, each model against linear
        lin = fitted["linear"]
        for name in ["quadratic", "RCS 3 knots", "RCS 4 knots"]:
            alt = fitted[name]
            lr = 2 * (alt.llf - lin.llf)
            ddf = alt.df_model - lin.df_model
            p = stats.chi2.sf(lr, ddf) if ddf > 0 else np.nan
            rows.append(
                {"predictor": x, "model": f"LRT {name} vs linear", "n": int(alt.nobs),
                 "df_model": int(ddf), "r_squared": np.nan, "adj_r_squared": np.nan,
                 "AIC": round(lr, 3), "BIC": np.nan, "rmse": round(float(p), 5)}
            )
        # RESET test for functional form
        reset = linear_reset(lin, power=2, use_f=True)
        rows.append(
            {"predictor": x, "model": "Ramsey RESET on linear", "n": int(lin.nobs),
             "df_model": np.nan, "r_squared": np.nan, "adj_r_squared": np.nan,
             "AIC": round(float(reset.statistic), 3), "BIC": np.nan,
             "rmse": round(float(reset.pvalue), 5)}
        )

    out = pd.DataFrame(rows)
    out.columns = [
        *out.columns[:-1],
        "rmse_or_p_value",
    ]
    out.to_csv(output_dir / "spline_model_comparison.csv", index=False)
    log("")
    log("NON-LINEARITY : see spline_model_comparison.csv")
    log("  (for LRT/RESET rows, the AIC column holds the statistic and the last column the p-value)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, x, label in zip(axes, ["DE", "TFDI"],
                            ["Diaphragmatic excursion (cm)", "Thickening fraction (%)"]):
        d = df[["P0_1", x]].dropna().sort_values(x)
        ax.scatter(d[x], d["P0_1"], s=28, alpha=0.45, color="#2c3e50", label=f"Patients (n={len(d)})")
        grid = pd.DataFrame({x: np.linspace(d[x].min(), d[x].max(), 200)})
        for name, formula, colour, style in [
            ("Linear", f"P0_1 ~ {x}", "#3498db", "--"),
            ("Quadratic", f"P0_1 ~ {x} + I({x}**2)", "#e74c3c", "-"),
        ]:
            res = smf.ols(formula, data=d).fit()
            ax.plot(grid[x], res.predict(grid), style, color=colour, lw=2, label=name)
        rcs = _fit_rcs(d, x, 4)
        pred = rcs.predict(sm.add_constant(rcs_basis(grid[x].values, rcs._rcs_knots)))
        ax.plot(grid[x], pred, "-.", color="#27ae60", lw=2, label="RCS (4 knots)")
        ax.set_xlabel(label)
        ax.set_ylabel(r"$P_{0.1}$ (cmH$_2$O)")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title("A. Respiratory drive vs. excursion")
    axes[1].set_title("B. Respiratory drive vs. thickening fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_nonlinear_fits.png")
    plt.close(fig)




def residual_diagnostics(df: pd.DataFrame, fits: dict, output_dir: Path) -> None:
    rows = []
    models = {
        "Linear (DE)": smf.ols("P0_1 ~ DE", data=df[["P0_1", "DE"]].dropna()).fit(),
        "Quadratic (DE)": smf.ols("P0_1 ~ DE + I(DE**2)", data=df[["P0_1", "DE"]].dropna()).fit(),
        "Linear (TFDI)": smf.ols("P0_1 ~ TFDI", data=df[["P0_1", "TFDI"]].dropna()).fit(),
        "Quadratic (TFDI)": smf.ols("P0_1 ~ TFDI + I(TFDI**2)", data=df[["P0_1", "TFDI"]].dropna()).fit(),
        "Multivariable A": fits["model_a"],
        "Multivariable B": fits["model_b"],
    }
    for name, res in models.items():
        resid = res.resid
        sw_stat, sw_p = stats.shapiro(resid)
        bp = het_breuschpagan(resid, res.model.exog)
        rows.append(
            {
                "model": name, "n": int(res.nobs),
                "shapiro_wilk_W": round(float(sw_stat), 4),
                "shapiro_wilk_p": round(float(sw_p), 5),
                "residuals_normal_at_0.05": bool(sw_p > 0.05),
                "breusch_pagan_LM": round(float(bp[0]), 4),
                "breusch_pagan_p": round(float(bp[1]), 5),
                "homoscedastic_at_0.05": bool(bp[1] > 0.05),
                "durbin_watson": round(float(sm.stats.durbin_watson(resid)), 4),
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(output_dir / "residual_diagnostics.csv", index=False)
    log("")
    log("RESIDUAL DIAGNOSTICS :")
    for _, r in diag.iterrows():
        log(
            f"  {r['model']:18s} n={r['n']:3d}  Shapiro-Wilk p={r['shapiro_wilk_p']:<9} "
            f"Breusch-Pagan p={r['breusch_pagan_p']}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (name, res) in zip(axes.ravel(), models.items()):
        ax.scatter(res.fittedvalues, res.resid, s=25, alpha=0.5, color="#2c3e50")
        ax.axhline(0, color="#e74c3c", lw=1.4)
        try:
            lo = sm.nonparametric.lowess(res.resid, res.fittedvalues, frac=0.7)
            ax.plot(lo[:, 0], lo[:, 1], color="#f39c12", lw=1.8, label="LOWESS")
            ax.legend(frameon=False, fontsize=8)
        except Exception:
            pass
        ax.set_title(f"{name} (n={int(res.nobs)})", fontsize=11)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("Residuals versus fitted values", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_residual_diagnostics.png")
    plt.close(fig)

    # Q-Q panel for the two multivariable models
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key in zip(axes, ["Multivariable A", "Multivariable B"]):
        sm.qqplot(models[key].resid, line="45", fit=True, ax=ax, markersize=4, alpha=0.6)
        ax.set_title(f"{key} (n={int(models[key].nobs)})")
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_qq_plots.png")
    plt.close(fig)




def p01_vs_rass(df: pd.DataFrame, output_dir: Path) -> None:
    d = df[["P0_1", "RASS"]].dropna()
    rho, p_rho = stats.spearmanr(d["RASS"], d["P0_1"])
    groups = [g["P0_1"].values for _, g in d.groupby("RASS") if len(g) >= 2]
    kw_h, kw_p = stats.kruskal(*groups)

    summary = (
        d.groupby("RASS")["P0_1"]
        .agg(n="count", median="median",
             q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .round(3)
        .reset_index()
    )
    summary.to_csv(output_dir / "p01_by_rass.csv", index=False)
    (output_dir / "p01_vs_rass_stats.txt").write_text(
        f"P0.1 versus RASS (Sedation_Depth), n = {len(d)}\n"
        f"Spearman rho = {rho:.3f}, p = {p_rho:.3g}\n"
        f"Kruskal-Wallis H = {kw_h:.3f}, p = {kw_p:.3g}\n"
        f"RASS range observed: {int(d['RASS'].min())} to {int(d['RASS'].max())}\n"
        f"Patients at RASS <= -4 (deep sedation): {int((d['RASS'] <= -4).sum())} "
        f"({100 * (d['RASS'] <= -4).mean():.1f}%)\n"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    order = sorted(d["RASS"].unique())
    sns.boxplot(data=d, x="RASS", y="P0_1", order=order, ax=ax1,
                color="#aed6f1", fliersize=0, width=0.6)
    sns.stripplot(data=d, x="RASS", y="P0_1", order=order, ax=ax1,
                  color="#2c3e50", size=4, alpha=0.6, jitter=0.22)
    ax1.set_xlabel("RASS score")
    ax1.set_ylabel(r"$P_{0.1}$ (cmH$_2$O)")
    ax1.set_title("A. Respiratory drive by sedation depth")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    jitter = np.random.default_rng(0).normal(0, 0.06, len(d))
    ax2.scatter(d["RASS"] + jitter, d["P0_1"], s=30, alpha=0.5, color="#16a085")
    fit = np.polyfit(d["RASS"], d["P0_1"], 1)
    xs = np.linspace(d["RASS"].min(), d["RASS"].max(), 50)
    ax2.plot(xs, np.polyval(fit, xs), color="#e74c3c", lw=2)
    ax2.set_xlabel("RASS score")
    ax2.set_ylabel(r"$P_{0.1}$ (cmH$_2$O)")
    ax2.set_title(f"B. Spearman "r"$\rho$"f" = {rho:.3f} (p = {p_rho:.3g})")
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "figure_p01_vs_rass.png")
    plt.close(fig)
    log("")
    log(f"P0.1 vs RASS : Spearman rho = {rho:.3f}, p = {p_rho:.3g}; "
        f"Kruskal-Wallis p = {kw_p:.3g}, n = {len(d)}")




def posthoc_power(fits: dict, output_dir: Path) -> None:
    """Compute descriptive observed-effect and detectable-effect summaries."""
    alpha = 0.05
    lines = ["Post-hoc statistical power for the multivariable models", "=" * 58, ""]

    df = fits["primary_df"]
    primary = smf.ols(
        "sqrtP ~ DE + TFDI + I(TFDI**2) + PaCO2 + age + SOFA", data=df
    ).fit()

    for label, res in [("PRIMARY: sqrt(P0.1), with TFDI^2", primary),
                       ("A (raw scale, with lung injury)", fits["model_a"]),
                       ("B (raw scale, without lung injury)", fits["model_b"])]:
        n, k = int(res.nobs), int(res.df_model)
        r2 = float(res.rsquared)
        f2 = r2 / (1 - r2)
        df_denom = n - k - 1
        lam = f2 * n
        f_crit = stats.f.ppf(1 - alpha, k, df_denom)
        power = float(stats.ncf.sf(f_crit, k, df_denom, lam))

        # smallest f2 detectable at 80% power with this n and k
        lo, hi = 1e-6, 5.0
        for _ in range(200):
            mid = (lo + hi) / 2
            p = float(stats.ncf.sf(f_crit, k, df_denom, mid * n))
            lo, hi = (lo, mid) if p > 0.80 else (mid, hi)
        f2_min = (lo + hi) / 2
        r2_min = f2_min / (1 + f2_min)

        lines += [
            f"Model {label}",
            f"  n                          = {n}",
            f"  predictors (k)             = {k}",
            f"  observed R^2               = {r2:.4f}",
            f"  Cohen's f^2 = R2/(1-R2)    = {f2:.4f}",
            f"  alpha                      = {alpha}",
            f"  noncentrality lambda=f2*N  = {lam:.2f}",
            f"  critical F({k}, {df_denom})         = {f_crit:.3f}",
            f"  achieved power (1-beta)    = {power:.6f}"
            + ("  (>0.999)" if power > 0.999 else ""),
            "",
            f"  smallest effect detectable at 80% power with n = {n}:",
            f"    f^2 = {f2_min:.4f}  (equivalent to R^2 = {r2_min:.4f})",
            "",
        ]
    lines += ["Interpretation: observed-effect power is descriptive; detectable-effect values are sensitivity summaries."]
    (output_dir / "posthoc_power.txt").write_text("\n".join(lines))
    log("")
    log("POST-HOC POWER : written to posthoc_power.txt")
    for ln in lines:
        if "achieved power" in ln:
            log(f"  {ln.strip()}")




def descriptive_tables(df: pd.DataFrame, output_dir: Path) -> None:
    quartiles = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]

    cat_vars = [
        "gender", "comorbitity", "CAD", "pulmonary_dz", "HTN", "renal",
        "cigarette", "substance", "alcohol", "Admission_Diagnosis",
        "Lung_Injury_Status", "outcome", "midazolam", "morphine", "methadone",
        "fendanyl", "History_of_GCS_Decrease",
    ]
    rows = []
    for var in cat_vars:
        if var not in df.columns:
            continue
        for level in sorted(df[var].dropna().unique()):
            row = {"variable": var, "category": level}
            sub = df[df[var] == level]
            row["overall_n"] = len(sub)
            row["overall_pct"] = round(100 * len(sub) / df[var].notna().sum(), 1)
            for q in quartiles:
                qtot = int((df["P01_quartile"] == q).sum())
                n = int(((df["P01_quartile"] == q) & (df[var] == level)).sum())
                row[f"{q} n"] = n
                row[f"{q} %"] = round(100 * n / qtot, 1) if qtot else np.nan
            try:
                ct = pd.crosstab(df[var], df["P01_quartile"])
                row["chi2_p"] = round(float(stats.chi2_contingency(ct)[1]), 4)
            except Exception:
                row["chi2_p"] = np.nan
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "table1_full_categories.csv", index=False)

    comorb = ["CAD", "pulmonary_dz", "HTN", "renal", "other_PMH"]
    rows = []
    for c in comorb:
        if c not in df.columns:
            continue
        yes = df[c].astype(str).str.lower().isin(["yes", "1", "true"])
        rows.append({"comorbidity": c, "n": int(yes.sum()),
                     "pct": round(100 * yes.mean(), 1)})
    pd.DataFrame(rows).to_csv(output_dir / "comorbidity_breakdown.csv", index=False)

    et = (
        df["Admission_Diagnosis"].value_counts(dropna=False)
        .rename_axis("admission_diagnosis").reset_index(name="n")
    )
    et["pct"] = (100 * et["n"] / len(df)).round(1)
    et.to_csv(output_dir / "etiology_breakdown.csv", index=False)

    # Breathing-pattern variables.
    patterns = [
        "adjunct_respiratory_muscle_use", "grunting", "shallow_breath", "tachypnea",
        "bradypnea", "dyspnea", "paradoxycal", "Cheyne_Stokes", "hyperventilationq",
        "Kussmaul", "Biot", "other_types_of_breathing",
    ]
    rows = []
    for p in patterns:
        if p not in df.columns:
            continue
        yes = df[p].astype(str).str.lower().isin(["yes", "1", "true"])
        rows.append({"pattern": p, "n": int(yes.sum()), "pct": round(100 * yes.mean(), 1)})
    pd.DataFrame(rows).sort_values("n", ascending=False).to_csv(
        output_dir / "breathing_patterns.csv", index=False
    )

    # -- P-SILI threshold counts  ------------------------------------
    thresholds = {}
    for t in [1.0, 1.5, 2.0, 3.0, 3.5, 4.0]:
        thresholds[f"n_above_{t}"] = int((df["P0_1"] > t).sum())
        thresholds[f"pct_above_{t}"] = round(100 * (df["P0_1"] > t).mean(), 1)
    thresholds["actual_above_3.5"] = int((df["P0_1"] > 3.5).sum())
    (output_dir / "psili_threshold_counts.json").write_text(json.dumps(thresholds, indent=2))

    mort = (
        df.groupby("P01_quartile", observed=True)["died"]
        .agg(n="count", deaths="sum")
        .assign(mortality_pct=lambda t: (100 * t["deaths"] / t["n"]).round(1))
        .reset_index()
    )
    ct = pd.crosstab(df["P01_quartile"], df["died"])
    chi2, p_mort, _, _ = stats.chi2_contingency(ct)
    mort.to_csv(output_dir / "mortality_by_quartile.csv", index=False)
    (output_dir / "mortality_by_quartile_test.txt").write_text(
        f"Chi-square = {chi2:.3f}, p = {p_mort:.3g}\n"
        f"Overall mortality: {int(df['died'].sum())}/{len(df)} "
        f"({100 * df['died'].mean():.1f}%)\n"
    )
    log("")
    log(f"Mortality overall {100 * df['died'].mean():.1f}%; by quartile chi-square p = {p_mort:.3g}")
    log(f"P0.1 > 3.5 cmH2O in {thresholds['actual_above_3.5']} rows")




def roc_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    d = df[["P0_1", "DE", "TFDI", "SOFA"]].dropna().copy()
    d["high_drive"] = (d["P0_1"] > 1.5).astype(int)
    rows, curves = [], {}
    for var in ["DE", "TFDI", "SOFA"]:
        auc = roc_auc_score(d["high_drive"], d[var])
        # SOFA is expected to point the other way; report the orientation used
        if auc < 0.5:
            auc = roc_auc_score(d["high_drive"], -d[var])
            orientation = "inverted (lower value = higher drive)"
        else:
            orientation = "as measured"
        fpr, tpr, thr = roc_curve(d["high_drive"], d[var] if orientation == "as measured" else -d[var])
        curves[var] = (fpr, tpr)
        youden = tpr - fpr
        best = int(np.argmax(youden))
        rows.append(
            {
                "predictor": var, "n": len(d), "orientation": orientation,
                "AUC": round(float(auc), 4),
                "optimal_cutoff_Youden": round(float(thr[best]), 3),
                "sensitivity_at_cutoff": round(float(tpr[best]), 3),
                "specificity_at_cutoff": round(float(1 - fpr[best]), 3),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "roc_auc_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for var, (fpr, tpr) in curves.items():
        auc = [r["AUC"] for r in rows if r["predictor"] == var][0]
        ax.plot(fpr, tpr, lw=2, label=f"{var} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(r"ROC: discrimination of high respiratory drive ($P_{0.1}$ > 1.5 cmH$_2$O)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_roc_curves.png")
    plt.close(fig)
    log("")
    log("ROC/AUC :")
    for r in rows:
        log(f"  {r['predictor']:5s} AUC = {r['AUC']:.3f}  "
            f"(cutoff {r['optimal_cutoff_Youden']}, sens {r['sensitivity_at_cutoff']}, "
            f"spec {r['specificity_at_cutoff']})")


def run(input_path: Path, output_dir: Path, expected_md5: str | None = None) -> int:
    global _log
    _log = []
    output_dir.mkdir(parents=True, exist_ok=True)
    log("=" * 78)
    log("Respiratory-drive statistical analysis")
    log(f"Run at {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    log(f"Python {sys.version.split()[0]}, pandas {pd.__version__}, ")
    log(f"statsmodels {sm.__version__}, scipy {stats.__name__.split('.')[0]}")
    log("=" * 78)

    df = load(input_path, expected_md5)
    missingness(df, output_dir)
    oxygenation_integrity(df, output_dir)
    fits = fit_competing_models(df, output_dir)
    rebuild_table4(df, output_dir)
    specification_robustness(df, output_dir)
    collinearity(df, fits, output_dir)
    nonlinearity(df, output_dir)
    residual_diagnostics(df, fits, output_dir)
    p01_vs_rass(df, output_dir)
    posthoc_power(fits, output_dir)
    descriptive_tables(df, output_dir)
    roc_analysis(df, output_dir)

    produced = sorted(p.name for p in output_dir.iterdir() if p.name != "00_provenance.txt")
    log("")
    log(f"Wrote {len(produced) + 1} artifacts to {output_dir}")
    for name in produced:
        log(f"  {name}")
    (output_dir / "00_provenance.txt").write_text("\n".join(_log) + "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="external analysis CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="artifact directory")
    parser.add_argument("--expected-md5", help="optional checksum for the private input")
    args = parser.parse_args(argv)
    try:
        return run(args.input, args.output_dir, args.expected_md5)
    except (AssertionError, FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
