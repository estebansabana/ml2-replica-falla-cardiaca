# Replicación con datos simulados: supervivencia en falla cardíaca

Trabajo final — Aprendizaje de Máquina II, Maestría en Analítica Aplicada, Universidad de La Sabana.

**Autor:** Esteban Bejarano Caldas · **Docente:** Jesús Antonio Villarraga Palomino

**Artículo replicado:** Chicco, D., & Jurman, G. (2020). *Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone.* BMC Medical Informatics and Decision Making, 20, 16. [https://doi.org/10.1186/s12911-020-1023-5](https://doi.org/10.1186/s12911-020-1023-5). Copia en `references/`.

---

## Resumen

El artículo concluye que **dos exámenes clínicos (creatinina sérica y fracción de eyección) bastan para predecir la muerte de pacientes con falla cardíaca**, igual o mejor que las 11 variables clínicas completas.

Este trabajo reconstruye el dataset **usando solo las estadísticas que publica el artículo**, repite su análisis y compara tres fuentes: lo que reporta el artículo, el dataset real y el dataset simulado.

**Resultado central:** con el dataset real se reproduce la tesis del artículo (Random Forest: MCC 0.404 con 2 variables frente a 0.368 con 11). Con los datos construidos a partir de sus propias tablas, **la tesis se invierte** (0.207 frente a 0.363). La conclusión depende de la forma de la distribución de la creatinina, que el artículo resume con solo tres números (mediana, media y desviación estándar) y que esos números no alcanzan a describir.

---

## Pregunta e hipótesis

**Pregunta:** ¿la información que publica el artículo basta para reproducir sus resultados?

Hipótesis registradas **antes** de entrenar los modelos, a partir del diagnóstico de fidelidad (la creatinina simulada pierde poder discriminante: AUC univariado 0.728 → 0.605):

| | Hipótesis | Resultado |
|---|---|---|
| **H1** | Con 2 variables, el MCC con datos simulados será menor que con datos reales | ✅ Confirmada en 3/3 modelos (p < 0.003) |
| **H2** | La brecha simulado − real será mayor con 2 variables que con 11 | ✅ Confirmada en 3/3 modelos |

---

## Correspondencia con la guía del curso

| Fase de la guía | Dónde está |
|---|---|
| 1–3. Artículo, estudio de los autores, características de los datos | Este README (Datos) · `references/` · `src/parametros.py` |
| 4. Simulación del conjunto de datos | `src/simulacion.py` · notebook 01 · README (Diseño de la simulación) |
| 5. Recreación del análisis | `src/modelos.py` · notebook 02 |
| 6. Comparación de resultados | Notebook 02 · README (Resultados) · `results/comparacion_*.csv` |
| 7. Extensión experimental | Notebook 03 · README (Extensión experimental) |
| 8. Conclusiones y aprendizajes | README (Conclusiones) |

---

## Datos

| Campo | Detalle |
|---|---|
| Cohorte | 299 pacientes con falla cardíaca (NYHA III–IV), Faisalabad, Pakistán, abril–diciembre de 2015 |
| Desenlace | `DEATH_EVENT`: muerte durante el seguimiento. 96 fallecidos (32.11 %), 203 sobrevivientes |
| Variables | 11 clínicas: edad, anemia, CPK, diabetes, fracción de eyección, hipertensión, plaquetas, creatinina sérica, sodio sérico, sexo, tabaquismo. Más el tiempo de seguimiento |
| Fuente real | [UCI Machine Learning Repository, dataset 519](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records); datos originales de Ahmad et al. (2017), *PLoS ONE* |
| Uso del dato real | **Solo validación**, con autorización del docente. Ningún parámetro del simulador sale de él |

---

## Estructura del repositorio

```
proyecto_ml_final/
├── README.md
├── requirements.txt            # dependencias mínimas
├── requirements_lock.txt       # versiones exactas usadas
├── data/
│   ├── raw/                    # dataset real UCI (solo validación)
│   └── simulated/              # datos generados por src/simulacion.py
├── notebooks/
│   ├── 01_simulacion_datos.ipynb        # simulación y fidelidad
│   ├── 02_analisis_articulo.ipynb       # réplica de las Tablas 4 y 9
│   └── 03_extension_experimental.ipynb  # tamaño de muestra
├── src/
│   ├── parametros.py           # Tablas 1–4 y 9 del artículo, transcritas y corregidas
│   ├── simulacion.py           # generador calibrado, condicionado a la clase
│   ├── fidelidad.py            # validación simulado vs. real vs. artículo
│   └── modelos.py              # réplica de los modelos y extensión experimental
├── results/                    # tablas de resultados (CSV) y bitácora
├── figures/                    # figuras generadas por los notebooks
└── references/                 # artículo en PDF
```

---

## Cómo reproducir

### 1. Entorno (macOS, zsh)

```zsh
cd proyecto_ml_final
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install libomp            # requerido por XGBoost en Apple Silicon
```

### 2. Simulación y validación desde la terminal (opcional, ~20 s)

```zsh
python -m src.simulacion
python -m src.fidelidad --simulado data/simulated/hf_simulado_n299_seed42.csv
```

Verificación de reproducibilidad: el archivo generado debe tener MD5 `a44d7ba686454bdbe7c900fb355a4e28`.

### 3. Notebooks, en orden

Abrir cada notebook en VS Code o Jupyter con el kernel `.venv` y ejecutar todas las celdas.

| Orden | Notebook | Tiempo aprox. (Apple Silicon) | Produce |
|---|---|---|---|
| 1 | `01_simulacion_datos` | 1–2 min | `figures/01_*.png`, `results/calibracion_distribuciones.csv`, `results/fidelidad_*.csv`, `results/auc_univariado_real_vs_sim.csv` |
| 2 | `02_analisis_articulo` | 5 min | `results/tabla4_*.csv`, `results/tabla9_*.csv`, comparaciones, pruebas y `figures/02_comparacion_mcc.png` |
| 3 | `03_extension_experimental` | 1–2 min | `results/extension_*.csv`, `results/diagnostico_naive_bayes_log.csv`, `figures/03_extension_tamano.png` |

Los notebooks localizan la raíz del repositorio automáticamente. Si se modifica un archivo de `src/`, hay que reiniciar el kernel.

Semilla: `random_state=42` en todos los procedimientos estocásticos.

---

## Diseño de la simulación

### Información disponible y no disponible

| Disponible en el artículo | No disponible |
|---|---|
| Rango de cada variable (Tabla 1) | Forma de las distribuciones (solo se sabe que no son normales: Shapiro-Wilk, Tabla 6) |
| Prevalencia de cada binaria **por clase** (Tabla 2) | Correlaciones entre predictores |
| Mediana, media y σ de cada numérica **por clase** (Tabla 3) | Granularidad de registro de cada variable |
| n = 299; 96 fallecidos (32.11 %) | Mecanismo de imputación de faltantes |

### Decisión 1 — Simulación condicionada a la clase, P(X | Y)

Se generan por separado 96 fallecidos y 203 sobrevivientes, cada grupo con los parámetros de su clase. Así la relación entre los predictores y el desenlace sale directamente de lo que reporta el artículo.

Se descartó generar primero los predictores y después el desenlace con un modelo logístico, P(Y | X), por dos razones: obliga a inventar coeficientes que el artículo no reporta, e impone una frontera lineal que favorece artificialmente a la regresión logística.

### Decisión 2 — Familia distribucional por variable

| Variable | Familia | Justificación |
|---|---|---|
| Edad | Gamma desplazada (soporte ≥ 40) | Cota inferior de inclusión; asimetría leve |
| CPK | Lognormal | Enzima positiva, media ≈ 2.5 × mediana: cola pesada |
| Fracción de eyección | Beta escalada a [14, 80] | Porcentaje acotado |
| Plaquetas | Gamma | Positiva, asimetría moderada |
| Creatinina sérica | Lognormal | Biomarcador positivo con cola derecha (falla renal) |
| Sodio sérico | Normal truncada | Media ≈ mediana: simétrica; el truncamiento reproduce las colas acotadas |
| Tiempo de seguimiento | Beta escalada a [4, 285] | Asimetría opuesta según la clase |
| Binarias | Bernoulli | Prevalencia de la Tabla 2 en cada clase |

### Decisión 3 — Calibración bajo truncamiento

Ajustar por momentos y después truncar al rango de la Tabla 1 desplaza los estadísticos: la media de la creatinina en sobrevivientes pasa de 1.19 a 1.35. Por eso los parámetros se calibran numéricamente (Nelder-Mead) para que la distribución **ya truncada** reproduzca la mediana, la media y la σ reportadas. Pérdida de calibración ≤ 0.04 en las 14 combinaciones de variable y clase.

### Decisión 4 — Independencia condicional a la clase

El código incluye una cópula gaussiana para introducir correlaciones (`correlaciones` en `simular()`), pero el escenario base usa independencia porque el artículo no publica correlaciones. Cualquier correlación tendría que venir de literatura clínica, **nunca del dataset real**, porque eso volvería circular la validación.

### Correcciones a lo publicado

- Tabla 2: anemia en sobrevivientes figura como 3; se usa 83 (consistente con el 40.89 % y con el total de 129).
- Tabla 9: las columnas F1 y Accuracy están intercambiadas. Se verifica con Accuracy = p·TPR + (1−p)·TNR.

---

## Fidelidad de la simulación

| Prueba | Resultado | Lectura |
|---|---|---|
| KS por variable y clase | 9 de 14 con p > 0.05 | Marginales aceptables; los fallos tienen causa identificada (cola de la CPK, forma de la creatinina, fracción de eyección redondeada a múltiplos de 5) |
| Proporciones binarias | \|diferencia\| ≤ 0.07 | Dentro del ruido muestral (error estándar ≈ 0.05 con n = 96) |
| Correlaciones (Spearman) | RMSE 0.12 | Se pierden relaciones que el artículo no reporta (p. ej., sexo–tabaquismo ρ ≈ 0.45 en el dato real) |
| AUC univariado de la creatinina | 0.728 real frente a 0.605 simulado (semilla 42; 0.63 en promedio de 50 semillas) | **Única variable relevante mal reproducida**. Fracción de eyección, edad y sodio: diferencia ≤ 0.013 |
| C2ST (clasificador real vs. simulado) | AUC 0.776 (piso real vs. real: 0.51) | Los datos son distinguibles, sobre todo por la forma de las continuas |

Además, el dataset real contiene **62 filas con valores imputados no declarados** en el artículo (p. ej., CPK = 582 en 47 pacientes; plaquetas = 263 358.03, la media exacta, en 25).

---

## Réplica del análisis

Protocolo del artículo, reimplementado en scikit-learn (`src/modelos.py`):

| Modelo del artículo (R) | Implementación | Hiperparámetros | Partición |
|---|---|---|---|
| Random Forest | `RandomForestClassifier`, 500 árboles | — | 80/20 |
| Decision tree | `DecisionTreeClassifier`, mínimos de `rpart` (20 / 7) | — | 80/20 |
| Gradient boosting | `XGBClassifier`, valores por defecto | — | 80/20 |
| Linear regression | Regresión lineal con umbral 0.5 | — | 80/20 |
| One rule | Árbol de profundidad 1 | — | 80/20 |
| Red neuronal | `MLPClassifier` | capas y neuronas, por MCC en validación | 60/20/20 |
| Naive Bayes | `GaussianNB` | — | 80/20 |
| SVM radial / lineal | `SVC` | C, por MCC en validación | 60/20/20 |
| kNN | `KNeighborsClassifier` | k, por MCC en validación | 60/20/20 |

- 100 particiones aleatorias por modelo, con semilla 42 + i **idéntica para el dato real y el simulado**, de modo que la única diferencia entre ambas corridas es el dataset.
- Tabla 9: Random Forest, XGBoost y SVM radial (C = 10) con creatinina y fracción de eyección; 70/30 estratificado.
- Umbral fijo de 0.5, como en el artículo. Estandarización dentro de `Pipeline` para los modelos basados en distancias.

---

## Resultados

### Tabla 9 del artículo: modelos con creatinina y fracción de eyección (MCC, 100 repeticiones)

| Modelo | Artículo | Dato real | Dato simulado | p real vs. simulado (Wilcoxon pareado) |
|---|---|---|---|---|
| Random Forest | 0.418 | 0.404 | 0.207 | 2.5 × 10⁻¹⁷ |
| Gradient Boosting | 0.414 | 0.395 | 0.174 | 2.0 × 10⁻¹⁷ |
| SVM radial | 0.348 | 0.338 | 0.291 | 2.3 × 10⁻³ |

### Hallazgos

| Hallazgo | Evidencia |
|---|---|
| La réplica del método es fiel | 7 de 10 modelos de la Tabla 4 y los 3 de la Tabla 9 quedan a ±0.04 de MCC del artículo con el dato real |
| El artículo probablemente no estandarizó las variables | Sin escalar, kNN da MCC −0.005 (artículo −0.025); con escalado, +0.162. Explica la discrepancia en kNN y la dirección del efecto en las SVM |
| La tesis del artículo no se sigue de sus propias tablas | Random Forest, dato real: 0.404 (2 var.) frente a 0.368 (11 var.), p = 0.049. Dato simulado: 0.207 frente a 0.363 |
| La evidencia del artículo para Random Forest es frágil | p = 0.049 en Apple Silicon y 0.055 en Linux x86, con las mismas versiones de librerías: la diferencia de plataforma basta para cruzar el umbral de 0.05. Además, las Tablas 4 y 9 usan particiones distintas, lo que confunde la comparación |
| La caída de ROC AUC del artículo no se reproduce | Random Forest con 2 variables: 0.792 (nuestro) frente a 0.698 (artículo); con 11 variables, 0.791 |
| La simulación favorece al modelo cuyos supuestos coinciden con el generador | Naive Bayes gana +0.10 de MCC con datos simulados (independencia condicional); los modelos de un solo corte pierden hasta −0.12 |

---

## Extensión experimental: tamaño de muestra

**Única condición modificada:** n = 299, 1 000 y 2 990 pacientes, con el mismo proceso generador (30 repeticiones, 70/30 estratificado). Responde a la principal limitación que declaran los autores.

| Random Forest (MCC) | n = 299 | n = 1 000 | n = 2 990 |
|---|---|---|---|
| 11 variables | 0.378 | 0.447 | 0.465 |
| 2 variables | 0.214 | 0.279 | 0.269 |
| σ entre repeticiones (11 var.) | 0.087 | 0.049 | 0.027 |

- La desviación cae en un factor 3.2 ≈ √10, como predice la teoría.
- El modelo de 2 variables **se estanca**: su techo lo fija cuánto se traslapan las clases en esas dos variables, y más datos no lo suben.
- SVM radial es la excepción: 2 variables ≥ 11 en todos los tamaños.
- Naive Bayes con 11 variables empeora con n (0.351 → 0.273) por suponer normalidad sobre variables lognormales. Con log(CPK) y log(creatinina) se mantiene en ≈ 0.35: es sesgo, no varianza.

Figura: `figures/03_extension_tamano.png`.

---

## Limitaciones

- **Una sola realización del dataset simulado** (semilla 42) para las Tablas 4 y 9. La variación entre semillas se evaluó solo en el diagnóstico de la creatinina (50 y 200 réplicas, notebook 01).
- **Independencia condicional supuesta.** Es el supuesto mínimo, pero favorece a Naive Bayes y elimina correlaciones clínicamente conocidas (creatinina–sodio, creatinina–edad).
- **Diferencias de implementación R → Python** no verificables: hiperparámetros por defecto de XGBoost, criterio de poda de `rpart`, posible ausencia de estandarización en el artículo. La estandarización explica bien kNN, pero solo parcialmente la SVM lineal.
- **Edad y fracción de eyección registradas con redondeo** en el dato real (múltiplos de 5). Ninguna distribución continua lo reproduce.
- **Umbral fijo de 0.5** en todas las métricas de matriz de confusión, para replicar el artículo. No es el umbral óptimo con clases desbalanceadas.

---

## Conclusiones

1. **Qué se logró recrear.** El protocolo del artículo, con fidelidad de ±0.04 de MCC en 10 de 13 modelos usando el dato real; la jerarquía de variables (fracción de eyección, edad y sodio) y la proporción de clases en los datos simulados.
2. **Qué no se logró recrear.** El poder discriminante de la creatinina, la variable principal del artículo, porque mediana, media y desviación no describen su forma; y las correlaciones entre variables, que el artículo no publica.
3. **Qué se aprendió del método.** Una simulación nunca es neutral: cada supuesto del generador favorece al modelo que comparte ese supuesto (independencia → Naive Bayes; un generador logístico habría favorecido a la regresión logística). Elegir la técnica de simulación es elegir qué comparación será justa.
4. **Cómo afecta la condición experimental.** Aumentar n reduce la varianza (σ ∝ 1/√n) pero no el sesgo: no rescata un modelo con poca información ni corrige un supuesto mal especificado.
5. **Implicación general.** Una conclusión que solo se verifica con los datos originales, y no con las estadísticas que se publican sobre ellos, es una conclusión frágil. Reportar únicamente tres estadísticos por variable no basta para que otros reproduzcan un hallazgo basado en la forma de una distribución.
6. **Implicaciones para el trabajo de grado** (predicción probabilística del perfil de interfases y recomendación del punto de corte en un poliducto multiproducto, con series SCADA):
    - **Modelar la distribución completa, no un resumen.** Aquí, tres estadísticos no reprodujeron la señal de la creatinina. En la tesis, predecir solo el volumen medio de interfase por par de productos perdería la forma del perfil, que es lo que decide el corte. Esto respalda la elección de regresión cuantílica e intervalos de predicción.
    - **No seleccionar modelos con datos generados por la física.** Si se simulan interfases con el modelo de dispersión axial de Austin–Palfrey mientras llegan los datos reales, ese generador favorecerá a los modelos que comparten su supuesto, igual que la independencia favoreció a Naive Bayes. La simulación sirve para construir y probar el pipeline; la comparación de modelos debe hacerse con datos SCADA reales.
    - **Tamaño de muestra comparable y métricas inestables.** Unas 480 interfases en 4 años es un orden de magnitud similar a los 299 pacientes del artículo. Hay que reportar intervalos de confianza y tamaños de efecto con validación por bloques temporales repetidos, no un único p-valor cerca de 0.05.
    - **Auditar artefactos del registro antes de modelar.** Los valores imputados y redondeados del dataset clínico tienen su equivalente en SCADA: valores congelados, interpolaciones e historización por banda muerta. Una prueba tipo C2ST o un análisis de distribuciones puede detectarlos.
    - **Separar estimación y decisión.** El umbral fijo de 0.5 del artículo es el análogo de una regla de corte fija. La tesis debe estimar el perfil con su incertidumbre y decidir el corte en una capa aparte, con pérdida asimétrica.
    - **Evitar la fuga de información.** El tiempo de seguimiento del dataset clínico codificaba el desenlace. En la tesis, el corte del operador y cualquier lectura posterior al punto de decisión no pueden usarse como variables ni como etiqueta.

---

## Índice de resultados

| Archivo en `results/` | Contenido |
|---|---|
| `bitacora.md` | Registro cronológico de cada experimento, decisión y hallazgo |
| `calibracion_distribuciones.csv` | Estadísticos de cada distribución calibrada frente a la Tabla 3 |
| `fidelidad_*.csv` | Marginales, binarias, diferencia de Spearman y ranking Mann-Whitney (simulado vs. real) |
| `auc_univariado_real_vs_sim.csv` | Poder discriminante de cada variable por separado |
| `tabla4_real_100rep.csv`, `tabla4_sim_100rep.csv` | Las 2 000 corridas de la Tabla 4 |
| `tabla9_real_100rep.csv`, `tabla9_sim_100rep.csv` | Las 600 corridas de la Tabla 9 |
| `comparacion_tabla4_MCC.csv`, `comparacion_tabla9_MCC.csv` | Artículo vs. real vs. simulado |
| `diagnostico_escalado.csv` | Experimento con y sin estandarización |
| `prueba_hipotesis_H1_H2.csv`, `significancia_tabla9.csv` | Verificación formal de las hipótesis |
| `extension_tamano_muestra.csv`, `extension_tamano_resumen.csv` | Extensión experimental |
| `diagnostico_naive_bayes_log.csv` | Naive Bayes con y sin transformación logarítmica, por tamaño de muestra |

---

## Referencias

- Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. *BMC Medical Informatics and Decision Making*, 20, 16. https://doi.org/10.1186/s12911-020-1023-5
- Ahmad, T., Munir, A., Bhatti, S. H., Aftab, M., & Raza, M. A. (2017). Survival analysis of heart failure patients: A case study. *PLoS ONE*, 12(7), e0181001.
- Heart Failure Clinical Records [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records
