import pandas as pd
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from config import SEASONS, USE_ALL_PLAYERS, LIMIT_PLAYERS
from tqdm import tqdm


def get_players():

    all_players = players.get_players()
    active = [p for p in all_players if p["is_active"]]

    return active if USE_ALL_PLAYERS else active[:LIMIT_PLAYERS]


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

    df = pd.concat(dfs, ignore_index=True)

    # -------------------------
    # CLEANING CRÍTICO
    # -------------------------

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["MATCHUP"] = df["MATCHUP"].str.upper()

    df["OPP"] = df["MATCHUP"].str.split(" ").str[-1]

    df["IS_HOME"] = df["MATCHUP"].str.contains(" VS ").astype(int)

    df = df.drop_duplicates()

    df.to_parquet("data/raw_games.parquet")

    print("Dataset guardado correctamente")


if __name__ == "__main__":
    download()