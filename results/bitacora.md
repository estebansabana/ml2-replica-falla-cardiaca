# Bitácora de experimentos

## 2026-10-14 — Simulación v1 (momentos sin truncar)

| Variable (sobrevivientes) | Objetivo (Tabla 3) | Simulado v1 |
|---|---|---|
| Creatinina sérica, media | 1.19 | 1.35 |
| Creatinina sérica, σ | 0.65 | 0.84 |

Causa: la distribución se ajusta por momentos y luego se trunca al rango de la Tabla 1 (mínimo 0.50 mg/dL). Ese truncamiento elimina ~12 % de la masa inferior y desplaza los momentos.
Corrección: calibración numérica (Nelder-Mead) de los parámetros, de modo que la distribución **truncada** reproduzca la mediana, la media y la σ de la Tabla 3.

## 2026-10-14 — Simulación v2 (calibrada)

Pérdida de calibración ≤ 0.04 en las 14 combinaciones de variable y clase (`reporte_calibracion()`).

Residuos estructurales:
- Edad de sobrevivientes: media (58.76) < mediana (60). La gamma es asimétrica a la derecha y no puede reproducir esto.
- CPK de fallecidos: la lognormal no puede ajustar a la vez la mediana (259) y la σ (1316).

### Validación contra el dataset real (UCI)

| Prueba | Resultado | Lectura |
|---|---|---|
| KS por variable y clase (semilla 42) | 9 de 14 con p > 0.05 | Las marginales se reproducen de forma aceptable |
| Proporciones binarias | \|dif\| ≤ 0.07 | Coinciden con la Tabla 2 dentro del ruido muestral |
| Spearman sim − real | RMSE 0.12; máx. 0.39 | Se pierde la dependencia entre variables (p. ej., sexo–tabaquismo, ρ = 0.45 en el dato real). El artículo no la reporta |
| AUC univariado de la creatinina | real 0.728 / sim 0.63 (media de 50 réplicas) | **Media, mediana y σ no bastan para reproducir la capacidad discriminante de la variable principal** |
| Ranking Mann-Whitney de la creatinina (200 réplicas) | Moda 2; rango 1–9 | Con n = 96 positivos el ranking es inestable |
| C2ST real vs. simulado | 0.776 (base); 0.759 (sin 62 filas imputadas); 0.709 (además, EF en múltiplos de 5) | Piso real vs. real = 0.51. La diferencia restante se atribuye a la forma de las distribuciones y a la dependencia ausente |

### Artefactos detectados en el dataset real (no declarados en el artículo)
- CPK = 582 en 47 pacientes, plaquetas = 263 358.03 en 25 y creatinina ∈ {1.18, 1.83} en 19: en total, 62 filas con valores imputados.
- Fracción de eyección registrada en solo 17 valores distintos, casi todos múltiplos de 5.

## 2026-10-18 — Verificación de reproducibilidad en el equipo local

- Entorno: macOS Apple Silicon, `.venv` creado desde `requirements.txt` (versiones exactas en `requirements_lock.txt`).
- `hf_simulado_n299_seed42.csv`: MD5 `a44d7ba686454bdbe7c900fb355a4e28`, idéntico al generado en el entorno de desarrollo (Linux, scikit-learn 1.9, scipy 1.18).
- Reporte de fidelidad idéntico; C2ST AUC = 0.776 ± 0.023.
- Conclusión: la simulación es reproducible bit a bit entre plataformas con semilla 42.

## 2026-10-18 — Notebook 02: réplica de las Tablas 4 y 9 (100 repeticiones, equipo local)

### Tabla 4 (MCC, 11 variables)
- 7 de 10 modelos quedan a ±0.04 del paper con el dato real: réplica del método validada.
- SVM radial (+0.130), SVM lineal (+0.243) y kNN (+0.179) superan al paper. Diagnóstico de un solo factor (30 rep.): sin estandarizar, kNN da MCC −0.005 (paper −0.025) y SVM lineal −0.031 (paper +0.107). Conclusión: el paper muy probablemente no estandarizó las variables; explica bien kNN y la dirección del efecto en SVM.
- Sim − real: pierden los modelos de corte único (árbol −0.101, One Rule −0.118, XGBoost −0.070); Random Forest ≈ 0; **Naive Bayes gana +0.102** porque la simulación cumple por construcción su supuesto de independencia condicional a la clase.

### Tabla 9 (MCC, creatinina + fracción de eyección)
| Modelo | Paper | Real | Simulado | p real vs sim (Wilcoxon pareado) |
|---|---|---|---|---|
| Random forests | 0.418 | 0.404 | 0.207 | 2.5e-17 |
| Gradient boosting | 0.414 | 0.395 | 0.174 | 2.0e-17 |
| SVM radial | 0.348 | 0.338 | 0.291 | 2.3e-03 |

- H1 (sim < real con 2 variables): confirmada en 3/3 modelos.
- H2 (brecha mayor con 2 que con 11 variables): confirmada en 3/3 modelos.
- Tesis del paper ("2 variables ≥ 11"): se reproduce en el dato real (RF 0.404 vs 0.368) pero se invierte en el simulado (RF 0.207 vs 0.363).
- Significancia de la tesis en el dato real, RF: Mann-Whitney p = 0.049 en el equipo local (0.055 en el entorno de desarrollo, con las mismas versiones de numpy, scipy, scikit-learn y xgboost: la diferencia proviene de la plataforma, Apple Silicon frente a Linux x86). Evidencia frágil, además confundida por el cambio de protocolo de partición entre T4 (80/20) y T9 (70/30 estratificado).
- ROC AUC con 2 variables (real): RF 0.792, frente a 0.698 del paper; con 11 variables, 0.791. La caída de AUC del paper no se reproduce.

## 2026-10-18 — Notebook 03: extensión experimental (tamaño de muestra)

Factor único: n ∈ {299, 1 000, 2 990}; 30 repeticiones 70/30 estratificadas; MCC medio (σ).

| Modelo | Variables | n = 299 | n = 1 000 | n = 2 990 |
|---|---|---|---|---|
| Random forests | 11 | 0.378 (0.087) | 0.447 (0.049) | 0.465 (0.027) |
| Random forests | 2 | 0.214 (0.077) | 0.279 (0.049) | 0.269 (0.030) |
| Gradient boosting | 11 | 0.280 (0.085) | 0.367 (0.048) | 0.418 (0.029) |
| Gradient boosting | 2 | 0.163 (0.095) | 0.306 (0.052) | 0.312 (0.029) |
| SVM radial | 11 | 0.258 (0.087) | 0.318 (0.052) | 0.375 (0.029) |
| SVM radial | 2 | 0.289 (0.085) | 0.371 (0.037) | 0.385 (0.026) |
| Naive bayes | 11 | 0.351 (0.091) | 0.305 (0.048) | 0.273 (0.030) |
| Naive bayes | 2 | 0.220 (0.076) | 0.268 (0.049) | 0.290 (0.030) |

- σ cae ≈ √10 al multiplicar n por 10, como predice la teoría.
- El modelo de 2 variables tiene un techo bajo en el mundo simulado: más datos no cierran la brecha con Random Forest ni con XGBoost.
- Diagnóstico Naive Bayes (30 rep.): con log(CPK) y log(creatinina), MCC 0.353 / 0.357 / 0.344 para n = 299 / 1 000 / 2 990. La caída con datos crudos es sesgo por normalidad mal especificada, no varianza.
