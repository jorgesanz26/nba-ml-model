import pandas as pd
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from config import SEASONS, USE_ALL_PLAYERS, LIMIT_PLAYERS, DATA_PATH
from tqdm import tqdm

def get_players():

    all_players = players.get_players()
    active = [p for p in all_players if p["is_active"]]

    if USE_ALL_PLAYERS:
        return active

    return active[:LIMIT_PLAYERS]


def download():

    player_list = get_players()

    dfs = []

    for p in tqdm(player_list, desc="Downloading NBA players"):
        pid = p["id"]

        for season in SEASONS:
            try:
                df = playergamelog.PlayerGameLog(
                    player_id=pid,
                    season=season
                ).get_data_frames()[0]

                df["PLAYER_ID"] = pid
                df["SEASON"] = season

                dfs.append(df)

                time.sleep(0.3)

            except:
                continue

    df = pd.concat(dfs)
    df.to_parquet(DATA_PATH)

    print("Dataset guardado en:", DATA_PATH)


if __name__ == "__main__":
    download()