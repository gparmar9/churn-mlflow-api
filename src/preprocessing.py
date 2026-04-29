import pathlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Rutas
PATH_PROCESSED = pathlib.Path(__file__).parent.parent / "data" / "processed"
PATH_INPUT     = PATH_PROCESSED / "churn_processed.csv"

# Columnas por tipo
numericas    = ["tenure", "monthlycharges"]
categoricas  = ["contract", "internetservice", "paymentmethod", "onlinesecurity", "techsupport"]
target        = "churn"


def load_data(path: pathlib.Path = PATH_INPUT) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[target] = (df[target] == "Yes").astype(int)
    return df


def build_pipeline() -> Pipeline:
    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ]
    )
    return Pipeline(steps=[("preprocesador", preprocesador)])


def split_and_save(
    df: pd.DataFrame,
    pipeline: Pipeline,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    X = df[numericas + categoricas]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_t = pipeline.fit_transform(X_train)
    X_test_t  = pipeline.transform(X_test)

    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_train_t).to_csv(PATH_PROCESSED / "X_train.csv", index=False)
    pd.DataFrame(X_test_t).to_csv(PATH_PROCESSED / "X_test.csv",  index=False)
    y_train.to_csv(PATH_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(PATH_PROCESSED / "y_test.csv",  index=False)

    print(f"Train: {X_train_t.shape} | Test: {X_test_t.shape}")
    return X_train_t, X_test_t, y_train, y_test


if __name__ == "__main__":
    df       = load_data()
    pipeline = build_pipeline()
    split_and_save(df, pipeline)