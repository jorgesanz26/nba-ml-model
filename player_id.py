from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd

df = leaguedashplayerstats.LeagueDashPlayerStats(timeout=60).get_data_frames()[0]

df_players = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]]

df_players.rename(columns={"TEAM_ABBREVIATION": "TEAM"}, inplace=True)

df_players.to_csv("data/players_with_team.csv", index=False)

print("OK")