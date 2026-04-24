import pandas as pd
import time
from nba_api.stats.endpoints import playergamelog

PLAYER_ID = 1628369  # Jayson Tatum
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

dfs = []

for season in SEASONS:
    gamelog = playergamelog.PlayerGameLog(
        player_id=PLAYER_ID,
        season=season
    )

    df = gamelog.get_data_frames()[0]
    df["PLAYER_ID"] = PLAYER_ID
    df["SEASON"] = season

    dfs.append(df)

    time.sleep(0.3)

df = pd.concat(dfs)

df.to_parquet("data/tatum_test.parquet")

print("OK - Tatum dataset creado:", df.shape)