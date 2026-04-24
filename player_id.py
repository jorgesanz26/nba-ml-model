from nba_api.stats.static import players

# Lista de todos los jugadores
nba_players = players.get_players()

# Buscar uno en concreto
player = [p for p in nba_players if p['full_name'] == 'LeBron James'][0]

print(player)