import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

def load_scores(path):
    with np.load(path) as data:
        return data["scores"]
    
def load_training_scores(folder):
    seeds = [42, 60000, 80000]
    training_scores = []

    for seed in seeds:
        path = f"{folder}/with_velocity_{seed}_0.01_0.99_training.npz"
        scores = load_scores(path)
        training_scores.append(scores)

    return np.stack(training_scores)

def calculate_learning_statistics(scores):
    blocks = scores.reshape(3, 150, 100)
    block_means = np.mean(blocks, axis=2)
    mean = np.mean(block_means, axis=0)
    std = np.std(block_means, axis=0)

    return mean, std

def plot_velocity_comparison():
    without_velocity = load_scores(
        "version1/results/without_velocity_50000_0.1_0.99_evaluation.npz"
    )

    with_velocity = load_scores(
        "version1/results/with_velocity_50000_0.1_0.99_evaluation.npz"
    )
    without_velocity_mean = np.mean(without_velocity)
    with_velocity_mean = np.mean(with_velocity)
    labels = ["Without velocity", "With velocity"]
    values = [without_velocity_mean, with_velocity_mean]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=["steelblue", "seagreen"])
    plt.bar_label(bars, fmt="%.2f", padding=3)
    plt.ylim(0, 1.1)
    plt.xlabel("State Representation")
    plt.ylabel("Average Score")
    plt.title("Effect of Velocity on Q-Learning Performance")
    plt.tight_layout()
    plt.savefig("figures/figure1_velocity.png", dpi=300)
    plt.close()

def plot_bar_chart(labels, values, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5))

    bars = plt.bar(labels, values, color="steelblue")
    plt.bar_label(bars, fmt="%.2f", padding=3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=300)
    plt.close()

def plot_learning_rates(values):
    plot_bar_chart(
        ["0.01", "0.1", "0.5"],
        values,
        "Effect of Learning Rate on Q-Learning Performance",
        "Learning Rate (Alpha)",
        "Average Score",
        "figure2_learning_rates"
    )

def plot_discretisation(values):
    plot_bar_chart(
        ["V1", "V2 Exp. 1", "V2 Exp. 2", "V2 Exp. 3"],
        values,
        "Effect of Discretisation and Training Configuration",
        "Experimental Configuration",
        "Average Validation Score",
        "figure3_discretisation"
    )

def plot_learning_curves(episodes, q_mean, q_std, sarsa_mean, sarsa_std):
    plt.figure(figsize=(10, 5))

    plt.plot(episodes, q_mean, color="steelblue", label="Q-learning")

    plt.fill_between(
        episodes,
        q_mean - q_std,
        q_mean + q_std,
        color="steelblue",
        alpha=0.15
    )

    plt.plot(episodes, sarsa_mean, color="darkorange", label="SARSA")

    plt.fill_between(
        episodes,
        sarsa_mean - sarsa_std,
        sarsa_mean + sarsa_std,
        color="darkorange",
        alpha=0.15
    )

    plt.xlabel("Training Episode")
    plt.ylabel("Average Score per 100 Episodes")
    plt.title("Q-Learning vs SARSA Training Performance")
    plt.legend(title="Shading: ±1 SD across seeds")
    plt.grid(alpha=0.2)
    plt.xlim(0, 15000)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("figures/figure4_learning_curves.png", dpi=300)
    plt.close()

def plot_final_comparison(labels, values):
    plt.figure(figsize=(10, 7))

    colors = (
        ["gray"] * 2
        + ["steelblue"] * 4
        + ["seagreen"] * 3
        + ["darkorange"] * 3
    )

    bars = plt.barh(labels, values, color=colors)
    plt.bar_label(bars, fmt="%.2f", padding=4)

    plt.gca().invert_yaxis()
    plt.xlim(0, max(values) * 1.2 if max(values) > 0 else 1)

    plt.xlabel("Average Score")
    plt.ylabel("Agent Configuration")
    plt.title("Final Independent Evaluation of AI Agents")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig("figures/figure5_final_comparison.png", dpi=300)
    plt.close()

def plot_coverage(labels, q_coverage, state_coverage):
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(11, 5))

    bars1 = plt.bar(
        x - width / 2,
        q_coverage,
        width,
        label="Non-zero Q-values",
        color="steelblue"
    )

    bars2 = plt.bar(
        x + width / 2,
        state_coverage,
        width,
        label="Non-zero states",
        color="seagreen"
    )

    plt.bar_label(bars1, fmt="%.1f%%", padding=3, fontsize=8)
    plt.bar_label(bars2, fmt="%.1f%%", padding=3, fontsize=8)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Coverage (%)")
    plt.title("Q-Table and State Coverage Comparison (Seed 42)")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig("figures/figure6_coverage.png", dpi=300)
    plt.close()

if __name__ == "__main__":

    # Figure 1 - Velocity comparison
    plot_velocity_comparison()

    # Figure 2 - Learning rate comparison
    plot_learning_rates([1.00, 0.80, 0.35])

    # Figure 3 - Discretisation experiments
    plot_discretisation([1.00, 0.29, 0.27, 0.66])

    # Figure 4 - Learning curves
    q_scores = load_training_scores("results_v2_exp3")
    sarsa_scores = load_training_scores("results_sarsa")

    q_mean, q_std = calculate_learning_statistics(q_scores)
    sarsa_mean, sarsa_std = calculate_learning_statistics(sarsa_scores)

    episodes = np.arange(100, 15001, 100)

    plot_learning_curves(
        episodes,
        q_mean,
        q_std,
        sarsa_mean,
        sarsa_std
    )
    # Figure 5 - Final independent evaluation
    labels = [
        "Uniform Random",
        "Biased Random",
        "V1 Without Velocity",
        "V1 Alpha 0.1",
        "V1 Alpha 0.01",
        "V1 Alpha 0.5",
        "V2 Q-Learning (42)",
        "V2 Q-Learning (60000)",
        "V2 Q-Learning (80000)",
        "SARSA (42)",
        "SARSA (60000)",
        "SARSA (80000)"
    ]

    values = [
        0.00, 0.03, 0.65, 0.80,
        1.00, 0.35, 0.72, 1.03,
        0.07, 0.07, 0.00, 0.25
    ]

    plot_final_comparison(labels, values)

    # Figure 6 - Q-table coverage (training seed 42)
    coverage_labels = [
        "V1 No Velocity",
        "V1 Alpha 0.01",
        "V1 Alpha 0.1",
        "V1 Alpha 0.5",
        "V2 Exp 1",
        "V2 Exp 2",
        "V2 Exp 3",
        "SARSA"
    ]

    q_coverage = [
        28.38, 11.14, 12.00, 13.20,
        8.32, 7.57, 10.31, 6.14
    ]

    state_coverage = [
        29.50, 13.83, 14.32, 15.25,
        11.20, 9.88, 13.28, 7.28
    ]

    plot_coverage(
        coverage_labels,
        q_coverage,
        state_coverage
    )

    print("Figures generated successfully.")