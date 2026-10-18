"""Parámetros de simulación extraídos de Chicco & Jurman (2020), Tablas 1-3.

Única fuente de información para el generador. El dataset real (data/raw)
NO se usa aquí; solo se emplea en src/fidelidad.py como referencia de validación.
"""

N_CLASE = {1: 96, 0: 203}  # DEATH_EVENT: fallecidos / sobrevivientes (Tabla 2)

# Tabla 1: soporte de cada variable
RANGOS = {
    "age": (40, 95),
    "creatinine_phosphokinase": (23, 7861),
    "ejection_fraction": (14, 80),
    "platelets": (25.01, 850.00),          # kiloplatelets/mL
    "serum_creatinine": (0.50, 9.40),
    "serum_sodium": (114, 148),
    "time": (4, 285),
}

# Tabla 3: (mediana, media, sigma) por clase
NUMERICAS = {
    1: {
        "age": (65.00, 65.22, 13.21),
        "creatinine_phosphokinase": (259.00, 670.20, 1316.58),
        "ejection_fraction": (30.00, 33.47, 12.53),
        "platelets": (258.50, 256.38, 98.53),
        "serum_creatinine": (1.30, 1.84, 1.47),
        "serum_sodium": (135.50, 135.40, 5.00),
        "time": (44.50, 70.89, 62.38),
    },
    0: {
        "age": (60.00, 58.76, 10.64),
        "creatinine_phosphokinase": (245.00, 540.10, 753.80),
        "ejection_fraction": (38.00, 40.27, 10.86),
        "platelets": (263.00, 266.66, 97.53),
        "serum_creatinine": (1.00, 1.19, 0.65),
        "serum_sodium": (137.00, 137.20, 3.98),
        "time": (172.00, 158.30, 67.74),
    },
}

# Tabla 2: proporción de "1" por clase.
# Corrección: la Tabla 2 reporta 3 sobrevivientes con anemia; el porcentaje de la
# misma tabla (40.89 %) y el total (129 - 46 = 83) implican 83. Se usa 83/203.
BINARIAS = {
    1: {"anaemia": 46 / 96, "diabetes": 40 / 96, "high_blood_pressure": 39 / 96,
        "sex": 62 / 96, "smoking": 30 / 96},
    0: {"anaemia": 83 / 203, "diabetes": 85 / 203, "high_blood_pressure": 66 / 203,
        "sex": 132 / 203, "smoking": 66 / 203},
}

# Familia distribucional por variable (justificación en README, sección Simulación)
FAMILIA = {
    "age": "gamma_desplazada",                  # soporte [40, inf), asimetría positiva leve
    "creatinine_phosphokinase": "lognormal",    # positiva, cola pesada (media >> mediana)
    "ejection_fraction": "beta_escalada",       # porcentaje acotado [14, 80]
    "platelets": "gamma",                       # positiva, asimetría moderada
    "serum_creatinine": "lognormal",            # positiva, cola pesada
    "serum_sodium": "normal",                   # media ~ mediana: simétrica, colas truncadas
    "time": "beta_escalada",                    # acotada [4, 285]; asimetría opuesta por clase
}

# Redondeo al formato de registro clínico
REDONDEO = {
    "age": 0,
    "creatinine_phosphokinase": 0,
    "ejection_fraction": 0,
    "platelets": 2,
    "serum_creatinine": 1,
    "serum_sodium": 0,
    "time": 0,
}

ORDEN_COLUMNAS = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
    "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
    "sex", "smoking", "time", "DEATH_EVENT",
]

# Resultados reportados por el artículo para la fase de comparación.
# Tabla 9: columnas F1 y Accuracy aparecen intercambiadas en el original
# (Accuracy = p*TPR + (1-p)*TNR = 0.754 para RF); aquí se reportan corregidas.
PAPER_TABLA4 = {
    "Random forests":            dict(MCC=0.384, F1=0.547, ACC=0.740, TPR=0.491, TNR=0.864, PR_AUC=0.657, ROC_AUC=0.800),
    "Decision tree":             dict(MCC=0.376, F1=0.554, ACC=0.737, TPR=0.532, TNR=0.831, PR_AUC=0.506, ROC_AUC=0.681),
    "Gradient boosting":         dict(MCC=0.367, F1=0.527, ACC=0.738, TPR=0.477, TNR=0.860, PR_AUC=0.594, ROC_AUC=0.754),
    "Linear regression":         dict(MCC=0.332, F1=0.475, ACC=0.730, TPR=0.394, TNR=0.892, PR_AUC=0.495, ROC_AUC=0.643),
    "One rule":                  dict(MCC=0.319, F1=0.465, ACC=0.729, TPR=0.383, TNR=0.892, PR_AUC=0.482, ROC_AUC=0.637),
    "Artificial neural network": dict(MCC=0.262, F1=0.483, ACC=0.680, TPR=0.428, TNR=0.815, PR_AUC=0.750, ROC_AUC=0.559),
    "Naive bayes":               dict(MCC=0.224, F1=0.364, ACC=0.696, TPR=0.279, TNR=0.898, PR_AUC=0.437, ROC_AUC=0.589),
    "SVM radial":                dict(MCC=0.159, F1=0.182, ACC=0.690, TPR=0.122, TNR=0.967, PR_AUC=0.587, ROC_AUC=0.749),
    "SVM linear":                dict(MCC=0.107, F1=0.115, ACC=0.684, TPR=0.072, TNR=0.981, PR_AUC=0.594, ROC_AUC=0.754),
    "k-nearest neighbors":       dict(MCC=-0.025, F1=0.148, ACC=0.624, TPR=0.121, TNR=0.866, PR_AUC=0.323, ROC_AUC=0.493),
}
PAPER_TABLA9 = {
    "Random forests":    dict(MCC=0.418, F1=0.585, ACC=0.754, TPR=0.541, TNR=0.855, PR_AUC=0.541, ROC_AUC=0.698),
    "Gradient boosting": dict(MCC=0.414, F1=0.585, ACC=0.750, TPR=0.550, TNR=0.845, PR_AUC=0.673, ROC_AUC=0.792),
    "SVM radial":        dict(MCC=0.348, F1=0.543, ACC=0.720, TPR=0.519, TNR=0.816, PR_AUC=0.494, ROC_AUC=0.667),
}
