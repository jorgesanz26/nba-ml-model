def print_best_picks(results, threshold=0.6):

    for player in results:
        best = []

        for stat, values in player["predictions"].items():
            for v in values:
                if v["prob_over"] >= threshold:
                    best.append((stat, v))

        if best:
            print(f"\n🔥 {player['player_name']}")

            for stat, v in best:
                print(
                    f"  {stat} > {v['line']} "
                    f"(prob: {v['prob_over']}, pred: {v['prediction']})"
                )