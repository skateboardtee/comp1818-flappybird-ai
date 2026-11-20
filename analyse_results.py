import numpy as np
from pathlib import Path

folders = [
    "results_v2_exp3",
    "results_sarsa"
]

for folder in folders:
    for path in sorted(Path(folder).glob("*_training.npz")):
        with np.load(path) as data:
            print("\n----------------------------------")
            print(path)
            print("----------------------------------")
            print("Episodes:", len(data["scores"]))
            print("Average reward (last 100):", np.mean(data["rewards"][-100:]))
            print("Average score (last 100):", np.mean(data["scores"][-100:]))
            print("Highest score:", np.max(data["scores"]))
            print("Average steps (last 100):", np.mean(data["steps"][-100:]))