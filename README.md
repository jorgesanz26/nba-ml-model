🏀 NBA Player Props Prediction Model

    Este proyecto construye un sistema de predicción de estadísticas de jugadores NBA (Points, Rebounds, Assists) y estima probabilidades de Over/Under usando Machine Learning.

📌 Objetivo

    Predecir estadísticas de jugadores NBA y evaluar si una línea de apuestas (betting line) tiene valor esperado positivo (EV+).

⚙️ Arquitectura del proyecto
    nba-model/
    │
    ├── data/                  # datasets procesados
    ├── models/                # modelos entrenados
    │
    ├── download_data.py       # descarga NBA API
    ├── build_features.py      # feature engineering
    ├── train_model.py         # entrenamiento ML
    ├── predict.py             # predicción
    ├── config.py              # configuración global
    │
    ├── print_results/         # output formateado
    └── README.md

🚀 Flujo completo del proyecto

    1️⃣ DESCARGA DE DATOS NBA

        📄 Ejecutar:

            python download_data.py

        🔧 Qué hace:

            Descarga partidos de NBA API
            Itera por jugadores activos
            Itera por temporadas definidas en config.py
            Guarda dataset crudo

        📁 Output:

            data/nba_dataset.parquet

        ⚠️ Configuración importante

            En config.py:
    
            SEASONS = ["2022-23", "2023-24", "2024-25"]
            USE_ALL_PLAYERS = True
            LIMIT_PLAYERS = 50  # para pruebas rápidas

    2️⃣ FEATURE ENGINEERING

        📄 Ejecutar:

            python build_features.py

        🔧 Qué hace:

            Crea variables de forma del jugador
            Rolling averages (últimos 5 partidos)
            Fatiga (minutos recientes)
            Codifica rival (oponente)

        📊 Features generadas:

            last5_pts
            last5_reb
            last5_ast
            last5_min
            fatigue
            opponent encoding

        📁 Output:

            data/nba_dataset.parquet (enriquecido)

    3️⃣ ENTRENAMIENTO DEL MODELO

        📄 Ejecutar:

            python train_model.py

        🧠 Modelo usado:

            XGBoost Regressor
            MultiOutput (PTS, REB, AST)

        🎯 Objetivo:

           Aprender relación entre:

           forma del jugador + contexto del partido → estadísticas finales

        📁 Output:

           models/nba_model.pkl

    4️⃣ PREDICCIÓN
    
        📄 Ejecutar:

            python predict.py

        🔧 Input requerido:

            player_id = 1628369  # Jayson Tatum
            opponent = "PHI"
            lines = {
                "PTS": 25.5,
                "REB": 6.5,
                "AST": 5.5
            }

        📊 Output ejemplo:

            🏀 JAYSON TATUM – PLAYER PROPS

        📊 PTS
        
            Prediction: 26.06
            Line: 25.5
            Over probability: 55.9% 🔼

        📊 REB

            Prediction: 5.29
            Line: 6.5
            Over probability: 36.7% 🔽

        📊 AST

            Prediction: 6.72
            Line: 5.5
            Over probability: 64.0% 🔼

    🔁 ORDEN DE EJECUCIÓN COMPLETO

        🟡 Primera vez (setup completo):

        python download_data.py
        python build_features.py
        python train_model.py

        🟢 Uso diario (predicción):

        python predict.py

⚙️ CONFIGURACIÓN GLOBAL

    Archivo:

    config.py

    Permite controlar:

    temporadas NBA
    número de jugadores
    rutas de datos
    comportamiento del pipeline

📦 Dependencias

    Instalar:

    pip install pandas numpy scikit-learn xgboost nba_api tqdm pyarrow joblib
    ⚡ Optimización de rendimiento
    Usa parquet en lugar de CSV
    Usa tqdm para ver progreso
    Cachea dataset (evitar re-descargas)
    No reentrenar modelo en cada ejecución

🧠 Notas importantes

    El modelo no se entrena en tiempo real
    Las predicciones dependen del dataset histórico
    El rendimiento puede variar por:
    lesiones
    cambios de rol
    matchup defensivo

📈 Futuras mejoras

    Incorporar odds reales de casas de apuestas
    Modelos bayesianos
    Ajuste por lesiones en tiempo real
    CLV (Closing Line Value)
    Dashboard interactivo

🏁 Resumen

Este proyecto permite:

✔ predecir stats NBA
✔ estimar over/under
✔ evaluar value bets
✔ construir pipeline cuantitativo completo