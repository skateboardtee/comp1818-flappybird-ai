import numpy as np
from pathlib import Path

for path in Path("results").glob("*_training.npz"):
    data = np.load(path)

    print("\n----------------------------------")
    print(path.name)
    print("----------------------------------")

    print("Episodes:", len(data["scores"]))
    print("Average reward (last 100):", np.mean(data["rewards"][-100:]))
    print("Average score (last 100):", np.mean(data["scores"][-100:]))
    print("Highest score:", np.max(data["scores"]))
    print("Average steps (last 100):", np.mean(data["steps"][-100:]))

    data.close()