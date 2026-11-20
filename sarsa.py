from q_learning import QLearningAgent

class SARSAAgent(QLearningAgent):

    def update(self, state, action, reward,
               next_state, next_action, terminated):

        current_q = self.q_table[state + (action,)]

        if terminated:
            future_q = 0
        else:
            future_q = self.q_table[next_state + (next_action,)]

        
        new_q = current_q + self.alpha * (
            reward + self.gamma * future_q - current_q
        )
        self.q_table[state + (action,)] = new_q
        
if __name__ == "__main__":
    agent = SARSAAgent(use_velocity=False)

    state = (5, 10)
    next_state = (6, 10)

    action = 1
    next_action = 0
    reward = 1

    agent.q_table[next_state] = [2.0, 5.0]

    print("Before update:", agent.q_table[state])

    agent.update(
        state, action, reward,
        next_state, next_action, False
    )

    print("After update:", agent.q_table[state])