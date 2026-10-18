"""Réplica del protocolo de clasificación de Chicco & Jurman (2020), Tablas 4 y 9.

Protocolo del artículo (implementado originalmente en R):
    Tabla 4 — 10 clasificadores, 11 variables clínicas, 100 particiones aleatorias.
        * Con hiperparámetros (red, SVM lineal, SVM radial, kNN): 60/20/20
          entrenamiento/validación/prueba; grid search que maximiza MCC en validación.
        * Sin hiperparámetros: 80/20 entrenamiento/prueba.
    Tabla 9 — RF, XGBoost y SVM radial con creatinina + fracción de eyección,
        100 particiones estratificadas 70/30.
    Umbral de decisión fijo en 0.5 para todas las métricas de matriz de confusión.

Equivalencias R -> scikit-learn y supuestos: ver README, sección "Réplica del análisis".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             matthews_corrcoef, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

OBJETIVO = "DEATH_EVENT"
VARIABLES_CLINICAS = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium", "sex", "smoking",
]
VARIABLES_TOP2 = ["serum_creatinine", "ejection_fraction"]


class RegresionLinealClasificador(BaseEstimator, ClassifierMixin):
    """Regresión lineal por mínimos cuadrados sobre y ∈ {0,1}, clasificada con umbral 0.5.

    Reproduce el 'Linear Regression' del artículo, que usa un modelo de regresión
    como clasificador binario.
    """

    def fit(self, X, y):
        self.modelo_ = LinearRegression().fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return self.modelo_.predict(X)

    def predict(self, X):
        return (self.decision_function(X) >= 0.5).astype(int)


def definir_modelos(random_state: int = 42) -> dict[str, tuple[object, dict | None]]:
    """Diccionario nombre -> (estimador, grilla de hiperparámetros o None)."""
    rs = random_state
    return {
        "Random forests": (RandomForestClassifier(n_estimators=500, random_state=rs, n_jobs=-1), None),
        "Decision tree": (DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=7, random_state=rs), None),
        "Gradient boosting": (XGBClassifier(eval_metric="logloss", random_state=rs, n_jobs=1), None),
        "Linear regression": (make_pipeline(StandardScaler(), RegresionLinealClasificador()), None),
        "One rule": (DecisionTreeClassifier(max_depth=1, random_state=rs), None),
        "Artificial neural network": (
            make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, random_state=rs)),
            {"mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50), (100, 100)]},
        ),
        "Naive bayes": (GaussianNB(), None),
        "SVM radial": (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=rs)),
            {"svc__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "SVM linear": (
            make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=rs)),
            {"svc__C": [0.001, 0.01, 0.1, 1, 10]},
        ),
        "k-nearest neighbors": (
            make_pipeline(StandardScaler(), KNeighborsClassifier()),
            {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21]},
        ),
    }


def puntaje(modelo, X) -> np.ndarray:
    """Puntaje continuo para curvas ROC/PR: probabilidad si existe, si no la función de decisión."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)


def metricas(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ACC": accuracy_score(y_true, y_pred),
        "TPR": recall_score(y_true, y_pred, zero_division=0),
        "TNR": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "PR_AUC": average_precision_score(y_true, y_score),
        "ROC_AUC": roc_auc_score(y_true, y_score),
    }


def ajustar_con_validacion(base, grilla, X_tr, y_tr, X_val, y_val):
    """Grid search manual: elige la combinación con mayor MCC en validación (como el artículo)."""
    mejor, mejor_mcc, mejor_params = None, -np.inf, None
    for params in ParameterGrid(grilla):
        m = clone(base).set_params(**params).fit(X_tr, y_tr)
        mcc = matthews_corrcoef(y_val, m.predict(X_val))
        if mcc > mejor_mcc:
            mejor, mejor_mcc, mejor_params = m, mcc, params
    return mejor, mejor_params


def replicar_tabla4(df: pd.DataFrame, n_rep: int = 100, random_state: int = 42,
                    modelos: list[str] | None = None, verbose: bool = True) -> pd.DataFrame:
    """Una fila por (repetición, modelo) con las 7 métricas sobre el conjunto de prueba."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X, y = df[VARIABLES_CLINICAS].to_numpy(float), df[OBJETIVO].to_numpy(int)
    catalogo = definir_modelos(random_state)
    nombres = modelos or list(catalogo)
    filas = []
    for rep in range(n_rep):
        semilla = random_state + rep
        X_tr80, X_te80, y_tr80, y_te80 = train_test_split(X, y, test_size=0.2, random_state=semilla)
        X_tmp, X_te60, y_tmp, y_te60 = train_test_split(X, y, test_size=0.2, random_state=semilla)
        X_tr60, X_val, y_tr60, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=semilla)
        for nombre in nombres:
            base, grilla = catalogo[nombre]
            if grilla is None:
                modelo, params, X_te, y_te = clone(base).fit(X_tr80, y_tr80), None, X_te80, y_te80
            else:
                modelo, params = ajustar_con_validacion(base, grilla, X_tr60, y_tr60, X_val, y_val)
                X_te, y_te = X_te60, y_te60
            fila = metricas(y_te, modelo.predict(X_te), puntaje(modelo, X_te))
            fila.update(rep=rep, modelo=nombre, params=str(params) if params else "")
            filas.append(fila)
        if verbose and (rep + 1) % 10 == 0:
            print(f"  repetición {rep + 1}/{n_rep}")
    return pd.DataFrame(filas)


def replicar_tabla9(df: pd.DataFrame, n_rep: int = 100, random_state: int = 42) -> pd.DataFrame:
    """RF, XGBoost y SVM radial con solo creatinina y fracción de eyección; 70/30 estratificado."""
    X, y = df[VARIABLES_TOP2].to_numpy(float), df[OBJETIVO].to_numpy(int)
    rs = random_state
    catalogo = {
        "Random forests": RandomForestClassifier(n_estimators=500, random_state=rs, n_jobs=-1),
        "Gradient boosting": XGBClassifier(eval_metric="logloss", random_state=rs, n_jobs=1),
        "SVM radial": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, random_state=rs)),
    }
    filas = []
    for rep in range(n_rep):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=rs + rep)
        for nombre, base in catalogo.items():
            modelo = clone(base).fit(X_tr, y_tr)
            fila = metricas(y_te, modelo.predict(X_te), puntaje(modelo, X_te))
            fila.update(rep=rep, modelo=nombre)
            filas.append(fila)
    return pd.DataFrame(filas)


METRICAS = ["MCC", "F1", "ACC", "TPR", "TNR", "PR_AUC", "ROC_AUC"]


def resumir(resultados: pd.DataFrame) -> pd.DataFrame:
    """Promedio por modelo (como reporta el artículo), ordenado por MCC."""
    return (resultados.groupby("modelo")[METRICAS].mean()
            .sort_values("MCC", ascending=False).round(3))


def comparar(paper: dict, real: pd.DataFrame, sim: pd.DataFrame, metrica: str = "MCC") -> pd.DataFrame:
    """Tabla de tres fuentes para una métrica: artículo, dato real (Python) y simulado (Python)."""
    tabla = pd.DataFrame({
        "paper": {m: v[metrica] for m, v in paper.items()},
        "real": real[metrica],
        "simulado": sim[metrica],
    })
    tabla["real - paper"] = tabla["real"] - tabla["paper"]
    tabla["sim - real"] = tabla["simulado"] - tabla["real"]
    return tabla.sort_values("paper", ascending=False).round(3)


def experimento_tamano(generador, factores: list[float], n_rep: int = 30,
                       random_state: int = 42) -> pd.DataFrame:
    """Extensión experimental: único factor variado = tamaño de muestra del proceso generador.

    Para cada factor se simula un dataset (misma semilla, misma proporción 32 % de positivos)
    y se evalúan los mismos modelos con 2 y con 11 variables bajo un protocolo común:
    particiones 70/30 estratificadas, n_rep repeticiones, umbral 0.5.
    """
    rs = random_state
    catalogo = {
        "Random forests": RandomForestClassifier(n_estimators=500, random_state=rs, n_jobs=-1),
        "Gradient boosting": XGBClassifier(eval_metric="logloss", random_state=rs, n_jobs=1),
        "SVM radial": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, random_state=rs)),
        "Naive bayes": GaussianNB(),
    }
    conjuntos = {"2 variables": VARIABLES_TOP2, "11 variables": VARIABLES_CLINICAS}
    filas = []
    for factor in factores:
        df = generador(factor_n=factor, random_state=rs)
        y = df[OBJETIVO].to_numpy(int)
        for nombre_conj, cols in conjuntos.items():
            X = df[cols].to_numpy(float)
            for rep in range(n_rep):
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y,
                                                          random_state=rs + rep)
                for nombre, base in catalogo.items():
                    modelo = clone(base).fit(X_tr, y_tr)
                    fila = metricas(y_te, modelo.predict(X_te), puntaje(modelo, X_te))
                    fila.update(n=len(df), variables=nombre_conj, modelo=nombre, rep=rep)
                    filas.append(fila)
        print(f"  n = {len(df)} listo")
    return pd.DataFrame(filas)
