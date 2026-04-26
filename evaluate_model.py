import pandas as pd
import joblib
from config import DATA_PATH, MODEL_PATH


def evaluate_model(df, model, lines_dict):

    correct = 0
    total = 0

    opp_cols = [c for c in df.columns if c.startswith("OPP_")]

    for _, row in df.iterrows():

        input_dict = {
            "last5_pts": row["last5_pts"],
            "last5_reb": row["last5_reb"],
            "last5_ast": row["last5_ast"],
            "last5_min": row["last5_min"],
            "fatigue": row["fatigue"]
        }

        for c in opp_cols:
            input_dict[c] = row[c]

        X = pd.DataFrame([input_dict])
        pred = model.predict(X)[0]

        real = [row["PTS"], row["REB"], row["AST"]]

        for i, stat in enumerate(["PTS", "REB", "AST"]):

            for line in lines_dict[stat]:

                pred_over = pred[i] > line
                real_over = real[i] > line

                if pred_over == real_over:
                    correct += 1

                total += 1

    return correct / total


if __name__ == "__main__":

    df = pd.read_parquet(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    LINES = {
        "PTS": [10, 15, 20, 25, 30],
        "REB": [5, 7, 10, 13],
        "AST": [5, 7, 10, 13]
    }

    acc = evaluate_model(df, model, LINES)

    print(f"Accuracy del modelo: {acc:.3f}")