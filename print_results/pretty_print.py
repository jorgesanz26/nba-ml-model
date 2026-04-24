def print_predictions(result, player_name="PLAYER"):


    print(f"🏀 {player_name.upper()} 🏀")


    for stat, data in result.items():

        pred = data["prediction"]
        line = data["line"]
        p_over = data["prob_over"]

        direction = "🔼 OVER" if p_over > 0.5 else "🔽 UNDER"

        print(f"📊 {stat}")
        print(f"   Prediction: {pred:.2f}")
        print(f"   Line: {line}")
        print(f"   Over probability: {p_over*100:.1f}% {direction}")
        print("-" * 40)