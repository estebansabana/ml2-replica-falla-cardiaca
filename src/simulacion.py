"""Generador de datos simulados condicionado a la clase (DEATH_EVENT).

Diseño:
    1. Cada variable continua se modela con una familia paramétrica ajustada a los
       estadísticos de la Tabla 3 (mediana, media, sigma) de su clase.
    2. La familia se trunca al rango de la Tabla 1 por transformada inversa
       restringida: u ~ U(F(a), F(b)), x = F^-1(u).
    3. Las binarias son Bernoulli con la prevalencia de la Tabla 2 en su clase.
    4. La dependencia entre variables se introduce con una cópula gaussiana
       (matriz R). Con R = I se obtiene independencia condicional a la clase,
       que es el único supuesto que el artículo permite sin información externa.

Uso (terminal, desde la raíz del repositorio):
    python -m src.simulacion                 # escenario base, n del artículo
    python -m src.simulacion --factor-n 10   # 2 990 registros, mismo proceso generador
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src import parametros as P

CONTINUAS = list(P.FAMILIA)
BINARIAS = list(P.BINARIAS[0])
VARIABLES = CONTINUAS + BINARIAS


def parametros_iniciales(variable: str, clase: int) -> np.ndarray:
    """Parámetros por método de momentos (sin truncar), en escala logarítmica cuando deben ser > 0."""
    mediana, media, sigma = P.NUMERICAS[clase][variable]
    a, b = P.RANGOS[variable]
    familia = P.FAMILIA[variable]
    if familia == "gamma_desplazada":
        m = media - a
        return np.log([(m / sigma) ** 2, sigma**2 / m])
    if familia == "gamma":
        return np.log([(media / sigma) ** 2, sigma**2 / media])
    if familia == "lognormal":
        # Punto de partida por mediana y media: la mediana es robusta a la cola pesada.
        return np.array([np.log(mediana), np.log(np.sqrt(2 * np.log(media / mediana)))])
    if familia == "beta_escalada":
        ancho = b - a
        m, v = (media - a) / ancho, (sigma / ancho) ** 2
        c = m * (1 - m) / v - 1
        return np.log([m * c, (1 - m) * c])
    if familia == "normal":
        return np.array([media, np.log(sigma)])
    raise ValueError(f"Familia no soportada: {familia}")


def construir(variable: str, theta: np.ndarray):
    """Distribución scipy congelada (sin truncar) a partir del vector de parámetros."""
    a, b = P.RANGOS[variable]
    familia = P.FAMILIA[variable]
    if familia == "gamma_desplazada":
        k, esc = np.exp(theta)
        return stats.gamma(a=k, loc=a, scale=esc)
    if familia == "gamma":
        k, esc = np.exp(theta)
        return stats.gamma(a=k, scale=esc)
    if familia == "lognormal":
        return stats.lognorm(s=np.exp(theta[1]), scale=np.exp(theta[0]))
    if familia == "beta_escalada":
        al, be = np.exp(theta)
        return stats.beta(a=al, b=be, loc=a, scale=b - a)
    if familia == "normal":
        return stats.norm(loc=theta[0], scale=np.exp(theta[1]))
    raise ValueError(f"Familia no soportada: {familia}")


def cuantil_truncado(dist, u: np.ndarray, a: float, b: float) -> np.ndarray:
    """Transformada inversa restringida al intervalo [a, b]."""
    fa, fb = dist.cdf(a), dist.cdf(b)
    return dist.ppf(fa + u * (fb - fa))


_U_GRID = (np.arange(4000) + 0.5) / 4000  # cuadratura determinista para momentos truncados
_CACHE: dict[tuple[str, int], object] = {}


def distribucion(variable: str, clase: int):
    """Distribución calibrada: sus estadísticos DESPUÉS de truncar a la Tabla 1
    reproducen mediana, media y sigma de la Tabla 3.

    Sin calibración, el truncamiento desplaza los momentos (p. ej., creatinina en
    sobrevivientes: media 1.35 frente a 1.19 reportado), lo que diluye la señal.
    """
    clave = (variable, clase)
    if clave in _CACHE:
        return _CACHE[clave]
    mediana, media, sigma = P.NUMERICAS[clase][variable]
    a, b = P.RANGOS[variable]

    def perdida(theta):
        try:
            x = cuantil_truncado(construir(variable, theta), _U_GRID, a, b)
        except (ValueError, FloatingPointError):
            return 1e6
        if not np.all(np.isfinite(x)):
            return 1e6
        return (((np.median(x) - mediana) / sigma) ** 2
                + ((x.mean() - media) / sigma) ** 2
                + ((x.std() - sigma) / sigma) ** 2)

    with np.errstate(all="ignore"):
        res = optimize.minimize(perdida, parametros_iniciales(variable, clase),
                                method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 4000})
    dist = construir(variable, res.x)
    dist.perdida_calibracion = float(res.fun)
    _CACHE[clave] = dist
    return dist


def reporte_calibracion() -> pd.DataFrame:
    """Estadísticos teóricos de cada distribución truncada calibrada frente a la Tabla 3."""
    filas = []
    for clase in (1, 0):
        for var in P.FAMILIA:
            a, b = P.RANGOS[var]
            d = distribucion(var, clase)
            x = cuantil_truncado(d, _U_GRID, a, b)
            med, mu, sd = P.NUMERICAS[clase][var]
            filas.append(dict(clase=clase, variable=var, familia=P.FAMILIA[var],
                              mediana_obj=med, mediana_cal=np.median(x),
                              media_obj=mu, media_cal=x.mean(), sd_obj=sd, sd_cal=x.std(),
                              perdida=d.perdida_calibracion))
    return pd.DataFrame(filas).round(3)


def matriz_correlacion(correlaciones: dict[tuple[str, str], float] | None) -> np.ndarray:
    """Construye R (cópula gaussiana) a partir de pares {(var_i, var_j): rho}."""
    R = np.eye(len(VARIABLES))
    if correlaciones:
        idx = {v: i for i, v in enumerate(VARIABLES)}
        for (vi, vj), rho in correlaciones.items():
            R[idx[vi], idx[vj]] = R[idx[vj], idx[vi]] = rho
    if np.min(np.linalg.eigvalsh(R)) <= 0:
        raise ValueError("La matriz de correlación no es definida positiva.")
    return R


def simular_clase(clase: int, n: int, R: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    z = rng.multivariate_normal(np.zeros(len(VARIABLES)), R, size=n)
    u = stats.norm.cdf(z)
    datos = {}
    for j, var in enumerate(VARIABLES):
        if var in CONTINUAS:
            a, b = P.RANGOS[var]
            x = cuantil_truncado(distribucion(var, clase), u[:, j], a, b)
            datos[var] = np.round(x, P.REDONDEO[var])
        else:
            datos[var] = (u[:, j] < P.BINARIAS[clase][var]).astype(int)
    df = pd.DataFrame(datos)
    df["DEATH_EVENT"] = clase
    return df


def simular(
    factor_n: float = 1.0,
    correlaciones: dict[tuple[str, str], float] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Genera el dataset completo.

    factor_n escala ambas clases manteniendo la proporción 96/203 (32.11 % positivos).
    """
    rng = np.random.default_rng(random_state)
    R = matriz_correlacion(correlaciones)
    partes = [simular_clase(c, int(round(n * factor_n)), R, rng) for c, n in P.N_CLASE.items()]
    df = pd.concat(partes, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df["platelets"] = df["platelets"] * 1000  # misma unidad que el CSV público (plaquetas/mL)
    for col in ["age", "creatinine_phosphokinase", "ejection_fraction", "serum_sodium", "time"]:
        df[col] = df[col].astype(int)
    return df[P.ORDEN_COLUMNAS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulación Chicco & Jurman (2020)")
    parser.add_argument("--factor-n", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--salida", type=Path, default=None)
    args = parser.parse_args()

    df = simular(factor_n=args.factor_n, random_state=args.seed)
    salida = args.salida or Path("data/simulated") / f"hf_simulado_n{len(df)}_seed{args.seed}.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"{len(df)} registros -> {salida}")
    print(df.groupby("DEATH_EVENT").size().to_dict())


if __name__ == "__main__":
    main()
