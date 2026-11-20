
import gymnasium as gym
import flappy_bird_gymnasium

# Create the game environment
env = gym.make(
    "FlappyBird-v0",
    render_mode="human",
    use_lidar=False
)

observation, info = env.reset(seed=42)

print("Observation:", observation)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Maximum episode steps:", env.spec.max_episode_steps)

# Test 3 episodes using random actions
for episode in range(10):
    observation, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        action = 1 if env.np_random.random() < 0.1 else 0
        if episode == 0 and steps < 10:
            print(
                f"Step {steps}: "
                f"Action={action}, "
                f"Bird Y={observation[9]:.3f}, "
                f"Velocity={observation[10]:.3f}"
            )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Final info:", info)
            break

    print(f"Episode {episode + 1}: Steps={steps}, Reward={total_reward:.2f}")

env.close()
