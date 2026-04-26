import pandas as pd
import numpy as np
import joblib

from config import DATA_PATH, MODEL_PATH

PLAYERS_PATH = "data/players_with_team.csv"


def simulate(pred, std=3.8):

    sims = np.random.normal(pred, std, 10000)

    # 🔻 añade sesgo negativo realista (NBA variance)
    shock = np.random.exponential(1.2, 10000)
    sims = sims - (shock * 0.6)

    return sims


# -------------------------
# FEATURE ENGINEERING
# -------------------------

def add_features(df):

    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    # días de descanso
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["rest_days"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(3)

    # back to back
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)

    # media temporada (rolling)
    df["season_avg_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["season_avg_reb"] = df.groupby("PLAYER_ID")["REB"].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df["season_avg_ast"] = df.groupby("PLAYER_ID")["AST"].transform(lambda x: x.rolling(20, min_periods=5).mean())

    # forma reciente
    df["last5_avg_pts"] = df.groupby("PLAYER_ID")["PTS"].transform(lambda x: x.rolling(5, min_periods=1).mean())

    df["form_trend"] = df["last5_avg_pts"] - df["season_avg_pts"]

    # minutos como proxy de uso
    df["usage_proxy"] = df.groupby("PLAYER_ID")["MIN"].transform(lambda x: x.rolling(10, min_periods=1).mean())

    return df


# -------------------------
# MAIN PREDICTION
# -------------------------

def predict_game(team, opponent, lines_dict):

    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    players_df = pd.read_csv(PLAYERS_PATH)

    df = df.merge(players_df, on="PLAYER_ID", how="left")

    # features nuevas
    df = add_features(df)

    df_team = df[df["TEAM"] == team]

    opp_cols = [c for c in df.columns if c.startswith("OPP_")]

    latest_players = (
        df_team.sort_values("GAME_DATE")
        .groupby("PLAYER_ID", as_index=False)
        .last()
    )

    results = []

    for _, last in latest_players.iterrows():

        input_dict = {

            # rendimiento reciente
            "last5_pts": last["last5_pts"],
            "last5_reb": last["last5_reb"],
            "last5_ast": last["last5_ast"],
            "last5_min": last["last5_min"],

            # fatiga
            "fatigue": last["fatigue"],
            "rest_days": last["rest_days"],
            "back_to_back": last["back_to_back"],

            # forma
            "season_avg_pts": last["season_avg_pts"],
            "season_avg_reb": last["season_avg_reb"],
            "season_avg_ast": last["season_avg_ast"],
            "form_trend": last["form_trend"],
            "usage_proxy": last["usage_proxy"],
        }

        # rival
        for c in opp_cols:
            input_dict[c] = 1 if c == f"OPP_{opponent}" else 0

        X = pd.DataFrame([input_dict])
        def shrink_prediction(pred, factor=0.88):
            return pred * factor

        pred = shrink_prediction(model.predict(X)[0], 0.88)

        player_result = {
            "player_id": last["PLAYER_ID"],
            "player_name": last["PLAYER_NAME"],
            "predictions": {}
        }

        for i, stat in enumerate(["PTS", "REB", "AST"]):

            sim = simulate(pred[i])

            player_result["predictions"][stat] = []

            for line in lines_dict[stat]:
                p_over = np.mean(sim > line) * 0.92
                p_over = min(p_over, 0.98)  # evita 1.0 artificiales

                player_result["predictions"][stat].append({
                    "line": line,
                    "prob_over": round(float(p_over), 3),
                    "prediction": round(float(pred[i]), 2)
                })

        results.append(player_result)

    return results


# -------------------------
# PRINT PICKS
# -------------------------

def print_best_picks(results, threshold=0.6):

    for player in results:

        picks = []

        for stat, values in player["predictions"].items():
            for v in values:
                if v["prob_over"] >= threshold:
                    picks.append((stat, v))

        if picks:
            print(f"\n🔥 {player['player_name']}")

            for stat, v in picks:
                print(
                    f"{stat} > {v['line']} | "
                    f"Prob: {v['prob_over']} | "
                    f"Pred: {v['prediction']}"
                )


# -------------------------
# RUN
# -------------------------

if __name__ == "__main__":

    LINES = {
        "PTS": [10, 15, 20, 25, 30],
        "REB": [5, 7, 10, 13],
        "AST": [5, 7, 10, 13]
    }

    team1 = "CLE"
    team2 = "TOR"

    results = predict_game(team1, team2, LINES)

    print_best_picks(results, threshold=0.6)