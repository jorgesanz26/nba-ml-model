import pandas as pd
import numpy as np
from config import DATA_PATH


def build():

    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    # targets
    df["y_pts"] = df["PTS"]
    df["y_reb"] = df["REB"]
    df["y_ast"] = df["AST"]

    # rolling stats
    df["last5_pts"] = df.groupby("PLAYER_ID")["PTS"].rolling(5).mean().reset_index(0, drop=True)
    df["last5_reb"] = df.groupby("PLAYER_ID")["REB"].rolling(5).mean().reset_index(0, drop=True)
    df["last5_ast"] = df.groupby("PLAYER_ID")["AST"].rolling(5).mean().reset_index(0, drop=True)

    df["last5_min"] = df.groupby("PLAYER_ID")["MIN"].rolling(5).mean().reset_index(0, drop=True)
    df["fatigue"] = df.groupby("PLAYER_ID")["MIN"].shift(1).rolling(5).mean().reset_index(0, drop=True)

    # opponent
    df["OPP"] = df["MATCHUP"].str.split(" ").str[-1]
    df = pd.get_dummies(df, columns=["OPP"], prefix="OPP")

    df = df.dropna()

    df.to_parquet(DATA_PATH)

    print("Features generadas")


if __name__ == "__main__":
    build()