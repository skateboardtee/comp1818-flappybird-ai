import numpy as np
from pathlib import Path

folders = [
    "results",
    "results_v2",
    "results_v2_exp2",
    "results_v2_exp3",
    "results_sarsa"
]

for folder in folders:
    for path in Path(folder).glob("*_q_table.npy"):
        q_table = np.load(path)

        print(f"\nModel: {path}")
        print("Shape:", q_table.shape)
        print("Total Q-values:", q_table.size)

        nonzero_values = np.count_nonzero(q_table)
        coverage = nonzero_values / q_table.size * 100

        print("Non-zero Q-values:", nonzero_values)
        print(f"Q-table coverage: {coverage:.2f}%")

        nonzero_states = np.count_nonzero(
            np.any(q_table != 0, axis=-1)
        )

        total_states = q_table.size // 2
        state_coverage = nonzero_states / total_states * 100

        print("Total states:", int(total_states))
        print("Non-zero states:", nonzero_states)
        print(f"State coverage: {state_coverage:.2f}%")
