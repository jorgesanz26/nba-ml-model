import pandas as pd
import joblib
from xgboost import XGBRegressor
from config import DATA_PATH, MODEL_PATH


def add_features(df):

    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["rest_days"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(3)
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)

    df["season_avg_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["season_avg_reb"] = df.groupby("PLAYER_ID")["REB"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["season_avg_ast"] = df.groupby("PLAYER_ID")["AST"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    df["last5_avg_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["form_trend"] = df["last5_avg_pts"] - df["season_avg_pts"]

    df["usage_proxy"] = df.groupby("PLAYER_ID")["MIN"].transform(lambda x: x.rolling(10, min_periods=1).mean())

    return df


# -------------------------
# DATA
# -------------------------

df = pd.read_parquet(DATA_PATH)
df = add_features(df)

opp_cols = [c for c in df.columns if c.startswith("OPP_")]

features = [
    "last5_pts",
    "last5_reb",
    "last5_ast",
    "last5_min",
    "fatigue",

    "rest_days",
    "back_to_back",

    "season_avg_pts",
    "season_avg_reb",
    "season_avg_ast",

    "form_trend",
    "usage_proxy",
] + opp_cols


X = df[features]
y = df[["PTS", "REB", "AST"]]


# -------------------------
# MODEL
# -------------------------

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)

print("Modelo XGBoost entrenado correctamente")