from nba_api.stats.static import players
import pandas as pd

# Obtener jugadores
nba_players = players.get_players()

data = []

for p in nba_players:
    full_name = p['full_name'].split()
    nombre = full_name[0]
    apellidos = " ".join(full_name[1:]) if len(full_name) > 1 else ""
    
    data.append({
        "Nombre": nombre,
        "Apellidos": apellidos,
        "ID": p['id']
    })

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar a CSV
df.to_csv("nba_players.csv", index=False, encoding="utf-8")

print("Archivo generado: nba_players.csv")