import numpy as np
from state import X_BINS, Y_BINS, V_BINS

class QLearningAgent:
    def __init__(self, use_velocity=True, alpha=0.1,
                 gamma=0.99, epsilon=1.0):

        self.use_velocity = use_velocity
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        if self.use_velocity == True:
            self.q_table = np.zeros((X_BINS, Y_BINS, V_BINS, 2))
        else:
            self.q_table = np.zeros((X_BINS, Y_BINS, 2))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state])
        
    def update(self, state, action, reward, next_state, terminated):
        current_q = self.q_table[state + (action,)]

        if terminated:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state])

        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        self.q_table[state + (action,)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


if __name__ == "__main__":
    agent = QLearningAgent(use_velocity=False)

    state = (5, 10)
    next_state = (6, 10)
    action = 1
    reward = 1

    print("Before update:", agent.q_table[state])

    agent.update(state, action, reward, next_state, False)

    print("After update:", agent.q_table[state])

    print("Epsilon before:", agent.epsilon)
    agent.decay_epsilon()
    print("Epsilon after:", agent.epsilon)