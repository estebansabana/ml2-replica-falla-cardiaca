"""Evaluación de fidelidad: datos simulados vs. dataset real (UCI) vs. artículo.

Niveles de validación:
    1. Marginales por clase   -> estadísticos + prueba KS de dos muestras.
    2. Binarias por clase     -> diferencia de proporciones.
    3. Dependencia            -> matrices de Spearman y su diferencia.
    4. Señal discriminante    -> ranking Mann-Whitney frente a la Tabla 5 del artículo.
    5. Indistinguibilidad     -> C2ST: clasificador real-vs-simulado (AUC ~ 0.5 = indistinguible).

Uso (terminal, desde la raíz del repositorio):
    python -m src.fidelidad --simulado data/simulated/hf_simulado_n299_seed42.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import parametros as P

RUTA_REAL = Path("data/raw/heart_failure_clinical_records_dataset.csv")
CONTINUAS = list(P.FAMILIA)
BINARIAS = list(P.BINARIAS[0])
PREDICTORAS = CONTINUAS[:-1] + BINARIAS  # sin 'time' (excluida del análisis principal del artículo)

PAPER_MW_RANKING = [
    "serum_creatinine", "ejection_fraction", "age", "serum_sodium", "high_blood_pressure",
    "anaemia", "platelets", "creatinine_phosphokinase", "smoking", "sex", "diabetes",
]


def cargar_real() -> pd.DataFrame:
    return pd.read_csv(RUTA_REAL)


def marginales(real: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clase in (1, 0):
        r, s = real[real.DEATH_EVENT == clase], sim[sim.DEATH_EVENT == clase]
        for var in CONTINUAS:
            escala = 1000 if var == "platelets" else 1
            med_p, mean_p, sd_p = P.NUMERICAS[clase][var]
            ks = stats.ks_2samp(r[var], s[var])
            filas.append({
                "clase": clase, "variable": var,
                "media_paper": mean_p, "media_real": r[var].mean() / escala, "media_sim": s[var].mean() / escala,
                "mediana_paper": med_p, "mediana_real": r[var].median() / escala, "mediana_sim": s[var].median() / escala,
                "sd_paper": sd_p, "sd_real": r[var].std() / escala, "sd_sim": s[var].std() / escala,
                "KS_D": ks.statistic, "KS_p": ks.pvalue,
            })
    return pd.DataFrame(filas).round(3)


def binarias(real: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for clase in (1, 0):
        r, s = real[real.DEATH_EVENT == clase], sim[sim.DEATH_EVENT == clase]
        for var in BINARIAS:
            filas.append({
                "clase": clase, "variable": var,
                "p_paper": P.BINARIAS[clase][var], "p_real": r[var].mean(), "p_sim": s[var].mean(),
            })
    df = pd.DataFrame(filas)
    df["dif_sim_real"] = df.p_sim - df.p_real
    return df.round(3)


def dependencia(real: pd.DataFrame, sim: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = PREDICTORAS + ["DEATH_EVENT"]
    rho_r = real[cols].corr(method="spearman")
    rho_s = sim[cols].corr(method="spearman")
    dif = rho_s - rho_r
    tri = np.triu_indices_from(dif, k=1)
    resumen = {
        "frobenius": float(np.linalg.norm(dif.values[tri])),
        "max_abs": float(np.abs(dif.values[tri]).max()),
        "rmse": float(np.sqrt(np.mean(dif.values[tri] ** 2))),
    }
    return dif.round(3), resumen


def ranking_mann_whitney(df: pd.DataFrame) -> pd.Series:
    p = {}
    for var in PREDICTORAS:
        g1, g0 = df.loc[df.DEATH_EVENT == 1, var], df.loc[df.DEATH_EVENT == 0, var]
        p[var] = stats.mannwhitneyu(g1, g0, alternative="two-sided").pvalue
    return pd.Series(p).sort_values()


def comparar_rankings(real: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    mw_r, mw_s = ranking_mann_whitney(real), ranking_mann_whitney(sim)
    df = pd.DataFrame({
        "rank_paper": {v: i + 1 for i, v in enumerate(PAPER_MW_RANKING)},
        "rank_real": {v: i + 1 for i, v in enumerate(mw_r.index)},
        "rank_sim": {v: i + 1 for i, v in enumerate(mw_s.index)},
        "p_real": mw_r, "p_sim": mw_s,
    }).sort_values("rank_paper")
    return df


def c2st(real: pd.DataFrame, sim: pd.DataFrame, random_state: int = 42) -> dict:
    """Classifier two-sample test: AUC de un RF que intenta separar real de simulado."""
    cols = PREDICTORAS + ["DEATH_EVENT"]
    X = pd.concat([real[cols], sim[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(real)), np.ones(len(sim))]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=random_state, n_jobs=-1)
    auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    return {"auc_media": float(auc.mean()), "auc_sd": float(auc.std()), "importancias": imp.round(3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulado", type=Path, required=True)
    parser.add_argument("--salida", type=Path, default=Path("results"))
    args = parser.parse_args()

    real, sim = cargar_real(), pd.read_csv(args.simulado)
    args.salida.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30)

    m = marginales(real, sim)
    b = binarias(real, sim)
    dif, res_dep = dependencia(real, sim)
    rk = comparar_rankings(real, sim)
    c2 = c2st(real, sim)

    m.to_csv(args.salida / "fidelidad_marginales.csv", index=False)
    b.to_csv(args.salida / "fidelidad_binarias.csv", index=False)
    dif.to_csv(args.salida / "fidelidad_dif_spearman.csv")
    rk.to_csv(args.salida / "fidelidad_ranking_mw.csv")

    print("== Marginales continuas ==\n", m.to_string(index=False))
    print("\n== Binarias ==\n", b.to_string(index=False))
    print("\n== Dependencia (Spearman sim - real) ==", res_dep)
    print("\n== Ranking Mann-Whitney ==\n", rk.to_string())
    print(f"\n== C2ST == AUC = {c2['auc_media']:.3f} ± {c2['auc_sd']:.3f}")
    print(c2["importancias"].to_string())


if __name__ == "__main__":
    main()
