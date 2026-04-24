import pandas as pd
import numpy as np
import joblib

from config import DATA_PATH, MODEL_PATH
from print_results.pretty_print import print_predictions


def simulate(pred, std=3.5):
    return np.random.normal(pred, std, 10000)


def predict(player_id, opponent, lines):

    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    df_p = df[df["PLAYER_ID"] == player_id]

    if df_p.empty:
        raise ValueError("Jugador no existe en dataset")

    last = df_p.iloc[-1]

    input_dict = {
        "last5_pts": last["last5_pts"],
        "last5_reb": last["last5_reb"],
        "last5_ast": last["last5_ast"],
        "last5_min": last["last5_min"],
        "fatigue": last["fatigue"]
    }

    opp_cols = [c for c in df.columns if c.startswith("OPP_")]

    for c in opp_cols:
        input_dict[c] = 1 if c == f"OPP_{opponent}" else 0

    X = pd.DataFrame([input_dict])

    pred = model.predict(X)[0]

    result = {}

    for i, stat in enumerate(["PTS", "REB", "AST"]):

        sim = simulate(pred[i])

        p_over = np.mean(sim > lines[stat])

        result[stat] = {
            "prediction": round(float(pred[i]), 2),
            "line": lines[stat],
            "prob_over": round(float(p_over), 3),
        }

    return result


if __name__ == "__main__":

    res = predict(
        player_id=2544,
        opponent="HOU",
        lines={"PTS": 20, "REB": 7, "AST": 7}
    )


print_predictions(res, player_name="Lebron James")