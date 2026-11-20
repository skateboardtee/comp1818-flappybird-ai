import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np

from state import get_state
from sarsa import SARSAAgent

def load_agent(use_velocity, seed=42, alpha=0.1, gamma=0.99):
    agent = SARSAAgent(
        use_velocity=use_velocity,
        epsilon=0.0
    )

    if use_velocity:
        name = "with_velocity"
    else:
        name = "without_velocity"

    agent.q_table = np.load(
        f"results_sarsa/{name}_{seed}_{alpha}_{gamma}_q_table.npy"
    )
    return agent

def evaluate(agent_type, episodes=100, seed=50000,
             alpha=0.1, gamma=0.99, model_seed=42):
    np.random.seed(seed)

    env = gym.make(
        "FlappyBird-v0",
        use_lidar=False,
        max_episode_steps=10000
    )

    rewards = []
    scores = []
    steps_history = []

    if agent_type == "with_velocity":
        agent = load_agent(True, alpha=alpha, gamma=gamma, seed=model_seed)
    elif agent_type == "without_velocity":
        agent = load_agent(False, alpha=alpha, gamma=gamma, seed=model_seed)
    else:
        agent = None

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)

        total_reward = 0
        steps = 0
        done = False

        while not done:

            if agent_type == "uniform_random":
                action = np.random.randint(0, 2)

            elif agent_type == "biased_random":
                action = 1 if np.random.random() < 0.1 else 0

            else:
                state = get_state(observation, agent.use_velocity)
                action = agent.choose_action(state)

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            done = terminated or truncated

        rewards.append(total_reward)
        scores.append(info["score"])
        steps_history.append(steps)

    env.close()

    print(f"\nAgent: {agent_type}")
    print(f"Training seed: {model_seed}")
    print(f"Evaluation seed: {seed}")
    print(f"Alpha: {alpha}")
    print(f"Gamma: {gamma}")
    print(f"Episodes: {episodes}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Average score: {np.mean(scores):.2f}")

    
    print(f"Highest score: {np.max(scores)}")
    print(f"Score standard deviation: {np.std(scores):.2f}")
    print(f"Average survival steps: {np.mean(steps_history):.2f}")

    success_rate = np.mean(np.array(scores) > 0) * 100
    print(f"Success rate: {success_rate:.2f}%")
    np.savez(
        f"results_sarsa/{agent_type}_{model_seed}_{seed}_{alpha}_{gamma}_evaluation.npz",
        rewards=rewards,
        scores=scores,
        steps=steps_history
    )

if __name__ == "__main__":
    for model_seed in [42, 60000, 80000]:
        evaluate("with_velocity", seed=50000,
                 alpha=0.01, model_seed=model_seed)
