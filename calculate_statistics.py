import numpy as np
from pathlib import Path

seeds = [42, 60000, 80000]

folders = {
    "Q-learning": "results_v2_exp3",
    "SARSA": "results_sarsa"
}

for algorithm, folder in folders.items():
    model_means = []

    for seed in seeds:
        path = Path(folder) / (
            f"with_velocity_{seed}_50000_0.01_0.99_evaluation.npz"
        )

        with np.load(path) as data:
            scores = data["scores"]

        average_score = np.mean(scores)
        model_means.append(average_score)

        print(f"{algorithm}, seed {seed}: {average_score:.4f}")

    model_means = np.array(model_means)

    print(f"\n{algorithm} summary")
    print(f"Mean score: {np.mean(model_means):.4f}")
    print(f"Sample SD: {np.std(model_means, ddof=1):.4f}")
    print(f"Population SD: {np.std(model_means):.4f}")
    print("-" * 40)
