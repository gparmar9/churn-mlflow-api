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
| Tracking | MLflow (Docker) |
| Visualizacion | matplotlib, seaborn |
| Contenedores | Docker |
| Entorno | venv |

---

## Estructura del proyecto

```
Proyecto 01 - Churn ML Flow/
│
├── data/
│   ├── raw/                    # Dataset original sin modificar
│   ├── processed/              # Splits de train/test generados por preprocessing.py
│   └── features/               # Features finales para entrenamiento
│
├── notebooks/
│   ├── 01_eda.ipynb            # Analisis exploratorio de datos (completado)
│   └── 02_modeling.ipynb       # Experimentacion con modelos
│
├── src/
│   ├── preprocessing.py        # Pipeline de carga, transformacion y split train/test
│   └── train.py                # Entrenamiento con MLflow tracking (pendiente)
│
├── mlruns/                     # Experimentos registrados por MLflow (autogenerado)
├── mlartifacts/                # Artefactos de MLflow (autogenerado)
│
├── models/                     # Modelos serializados
│
├── api/                        # API de inferencia FastAPI (pendiente)
├── docker/                     # Dockerfiles adicionales
│
├── docs/
│   └── ml_roadmap.html         # Roadmap del proyecto
│
├── tests/                      # Tests unitarios
│
├── docker-compose.yml          # Servicio MLflow
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Fases del proyecto

- [x] **Fase 1 — EDA**: analisis exploratorio, distribucion de variables, tasa de churn
- [x] **Fase 2 — Preprocesamiento**: encoding, escalado, manejo de nulos, split train/test
- [ ] **Fase 3 — Modelado**: entrenamiento de modelos (Logistic Regression, Random Forest, XGBoost)
- [ ] **Fase 4 — MLflow Tracking**: registro de experimentos, parametros, metricas y artefactos
- [ ] **Fase 5 — Evaluacion**: comparacion de modelos, curva ROC, matriz de confusion
- [ ] **Fase 6 — Model Registry**: registro del mejor modelo en MLflow Model Registry
- [ ] **Fase 7 — API + Deploy**: endpoint FastAPI containerizado con Docker

---

## Dataset

**Telco Customer Churn** — contiene informacion de ~7000 clientes de una empresa de telecomunicaciones.

Fuente: [IBM Sample Datasets / Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

> El archivo `data/raw/Telco-Churn.csv` no se versiona en este repositorio.
> Descargarlo desde el enlace anterior y colocarlo en `data/raw/`.

Features seleccionadas tras el EDA:

| Feature | Tipo | Motivo |
|---|---|---|
| `tenure` | numerica | Clientes nuevos churnan significativamente mas |
| `monthlycharges` | numerica | Cargos altos correlacionan con mayor churn |
| `contract` | categorica | Variable con mayor poder discriminativo del dataset |
| `internetservice` | categorica | Fiber optic concentra tasa de churn muy alta |
| `paymentmethod` | categorica | Electronic check destaca sobre el resto |
| `onlinesecurity` | categorica | Sin seguridad online = mayor probabilidad de churn |
| `techsupport` | categorica | Sin soporte tecnico = mayor probabilidad de churn |

Target: `churn` (0 = No, 1 = Yes) — desbalanceo 74/26%, se compensa en el modelado.

---

## Inicio rapido

```bash
# Crear entorno e instalar dependencias
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# Ejecutar pipeline de preprocesamiento
python src/preprocessing.py

# Levantar MLflow con Docker
docker-compose up
# Abrir http://localhost:5000
```

---

## Metricas objetivo

| Metrica | Objetivo |
|---|---|
| ROC-AUC | >= 0.85 |
| Recall (churn=1) | >= 0.75 |
| Precision (churn=1) | >= 0.70 |
