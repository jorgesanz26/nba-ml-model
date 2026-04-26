import pandas as pd
import numpy as np
from config import DATA_PATH


def build():

    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # -------------------------
    # TARGETS
    # -------------------------
    df["y_pts"] = df["PTS"]
    df["y_reb"] = df["REB"]
    df["y_ast"] = df["AST"]

    # -------------------------
    # ROLLING STATS (base features)
    # -------------------------
    df["last5_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["last5_reb"] = df.groupby("PLAYER_ID")["REB"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["last5_ast"] = df.groupby("PLAYER_ID")["AST"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["last5_min"] = df.groupby("PLAYER_ID")["MIN"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    # -------------------------
    # FATIGA
    # -------------------------
    df["rest_days"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(3)
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)

    df["fatigue"] = df.groupby("PLAYER_ID")["MIN"].shift(1).rolling(5, min_periods=1).mean()

    # -------------------------
    # FORM / CONTEXTO
    # -------------------------
    df["season_avg_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["season_avg_reb"] = df.groupby("PLAYER_ID")["REB"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["season_avg_ast"] = df.groupby("PLAYER_ID")["AST"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    df["last5_avg_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["form_trend"] = df["last5_avg_pts"] - df["season_avg_pts"]

    df["usage_proxy"] = df.groupby("PLAYER_ID")["MIN"].transform(lambda x: x.rolling(10, min_periods=1).mean())

    # -------------------------
    # OPPONENT
    # -------------------------
    df["OPP"] = df["MATCHUP"].str.split(" ").str[-1]
    df = pd.get_dummies(df, columns=["OPP"], prefix="OPP")

    # -------------------------
    # CLEAN
    # -------------------------
    df = df.dropna()

    # 🔥 IMPORTANTE: no sobrescribir raw dataset
    df.to_parquet("data/features.parquet")

    print("Features generadas correctamente")