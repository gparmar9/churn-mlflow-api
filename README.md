# Proyecto 01 — Churn Prediction con MLflow

Prediccion de abandono de clientes (churn) sobre el dataset Telco Customer Churn, con seguimiento completo del ciclo de vida del modelo usando MLflow.

---

## Objetivo

Construir un pipeline de Machine Learning end-to-end que prediga si un cliente va a darse de baja, rastreando experimentos, parametros y metricas con MLflow.

---

## Stack

| Capa | Tecnologia |
|---|---|
| Lenguaje | Python 3.12 |
| Datos | pandas, numpy |
| Modelado | scikit-learn |
| Tracking | MLflow |
| Visualizacion | matplotlib, seaborn |
| Entorno | venv |

---

## Estructura del proyecto

```
Proyecto 01 - Churn ML Flow/
│
├── data/
│   ├── raw/                    # Dataset original sin modificar
│   │   └── Telco-Churn.csv
│   ├── processed/              # Datos limpios y transformados
│   └── features/               # Features finales para entrenamiento
│
├── notebooks/
│   ├── 01_eda.ipynb            # Analisis exploratorio de datos
│   ├── 02_preprocessing.ipynb  # Limpieza y feature engineering
│   └── 03_modeling.ipynb       # Experimentacion con modelos
│
├── src/
│   ├── data/
│   │   ├── load.py             # Carga y validacion del dataset
│   │   └── preprocess.py       # Pipeline de preprocesamiento
│   ├── features/
│   │   └── build_features.py   # Construccion de features
│   ├── models/
│   │   ├── train.py            # Entrenamiento con MLflow tracking
│   │   ├── evaluate.py         # Metricas y evaluacion
│   │   └── predict.py          # Inferencia sobre nuevos datos
│   └── utils/
│       └── helpers.py
│
├── mlruns/                     # Experimentos registrados por MLflow (autogenerado)
│
├── models/                     # Modelos serializados (.pkl / MLflow artifacts)
│
├── docs/
│   └── ml_roadmap.html         # Roadmap del proyecto
│
├── tests/                      # Tests unitarios
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Fases del proyecto

- [ ] **Fase 1 — EDA**: analisis exploratorio, distribucion de variables, tasa de churn
- [ ] **Fase 2 — Preprocesamiento**: encoding, escalado, manejo de nulos, balanceo de clases
- [ ] **Fase 3 — Feature Engineering**: seleccion y construccion de features relevantes
- [ ] **Fase 4 — Modelado**: entrenamiento de modelos base (Logistic Regression, Random Forest, XGBoost)
- [ ] **Fase 5 — MLflow Tracking**: registro de experimentos, parametros, metricas y artefactos
- [ ] **Fase 6 — Evaluacion**: comparacion de modelos, curva ROC, matriz de confusion
- [ ] **Fase 7 — Model Registry**: registro del mejor modelo en MLflow Model Registry

---

## Dataset

**Telco Customer Churn** — contiene informacion de ~7000 clientes de una empresa de telecomunicaciones.

Fuente: [IBM Sample Datasets / Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> El archivo `data/raw/Telco-Churn.csv` no se versiona en este repositorio.
> Descargarlo desde el enlace anterior y colocarlo en `data/raw/`.

Variables clave:
- Datos demograficos: genero, edad, dependientes
- Servicios contratados: telefono, internet, streaming
- Cuenta: tipo de contrato, metodo de pago, cargos mensuales
- Target: `Churn` (Yes / No)

---

## Inicio rapido

```bash
# Clonar y crear entorno
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Lanzar MLflow UI
mlflow ui
# Abrir http://localhost:5000
```

---

## Metricas objetivo

| Metrica | Objetivo |
|---|---|
| ROC-AUC | >= 0.85 |
| Recall (churn=1) | >= 0.75 |
| Precision (churn=1) | >= 0.70 |
