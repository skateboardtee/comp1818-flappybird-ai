import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np

from state import get_state
from q_learning import QLearningAgent
def train(use_velocity=True, episodes=5000, seed=42,
          alpha=0.1, gamma=0.99):
    print(f"\nStarting training with seed {seed}")
    np.random.seed(seed)
    env = gym.make(
        "FlappyBird-v0",
        use_lidar=False,
        max_episode_steps=10000
    )
    agent = QLearningAgent(use_velocity=use_velocity, alpha=alpha, gamma=gamma)

    rewards = []
    scores = []
    steps_history = []

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        state = get_state(observation, use_velocity)

        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)

            observation, reward, terminated, truncated, info = env.step(action)

            next_state = get_state(observation, use_velocity)
            
            agent.update(state, action, reward, next_state, terminated)

            state = next_state

            total_reward += reward
            steps += 1

            done = terminated or truncated

        rewards.append(total_reward)
        scores.append(info["score"])
        steps_history.append(steps)

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}/{episodes}")
            print(f"Average reward: {np.mean(rewards[-100:]):.2f}")
            print(f"Average score: {np.mean(scores[-100:]):.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")

    env.close()

    if use_velocity:
        name = "with_velocity"
    else:
        name = "without_velocity"


    np.save(f"results_v2_exp3/{name}_{seed}_{alpha}_{gamma}_q_table.npy", agent.q_table)
    np.savez(
        f"results_v2_exp3/{name}_{seed}_{alpha}_{gamma}_training.npz",
        rewards=rewards,
        scores=scores,
        steps=steps_history
    )

if __name__ == "__main__":
    train(use_velocity=True, episodes=15000, seed=60000, alpha=0.01)
    train(use_velocity=True, episodes=15000, seed=80000, alpha=0.01)