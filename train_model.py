import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from config import DATA_PATH, MODEL_PATH


def train():

    df = pd.read_parquet(DATA_PATH)

    features = [
        "last5_pts",
        "last5_reb",
        "last5_ast",
        "last5_min",
        "fatigue"
    ] + [c for c in df.columns if c.startswith("OPP_")]

    X = df[features]
    y = df[["y_pts", "y_reb", "y_ast"]]

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5
        )
    )

    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)

    print("Modelo guardado")


if __name__ == "__main__":
    train()