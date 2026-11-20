import gymnasium as gym
import flappy_bird_gymnasium

from evaluate import load_agent
from state import get_state

def watch_agent(alpha=0.01, episodes=5, seed=40000):
    env = gym.make(
        "FlappyBird-v0",
        render_mode="human",
        use_lidar=False,
        max_episode_steps=10000
    )

    agent = load_agent(True, alpha=alpha)

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        done = False

        while not done:
            state = get_state(observation, agent.use_velocity)
            action = agent.choose_action(state)

            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

        print(f"Episode {episode + 1}: Score = {info['score']}")

    env.close()

if __name__ == "__main__":
    watch_agent(alpha=0.01)
    watch_agent(alpha=0.1)
    watch_agent(alpha=0.5)