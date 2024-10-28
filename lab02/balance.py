import math
import numpy as np
import gymnasium as gym  # Changed from 'gym' to 'gymnasium'


class QLearner:
    def __init__(self):
        self.environment = gym.make('CartPole-v1', render_mode="human")
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        
        # Add Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0  # Start with 100% exploration
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.n_bins = 6  # Number of bins for each observation
        self.q_table = {}  # Initialize empty Q-table

    def learn(self, max_attempts):
        for _ in range(max_attempts):
            reward_sum = self.attempt()
            print(reward_sum)

    def attempt(self):
        # Unpack the observation from reset() return value
        observation, _ = self.environment.reset()
        observation = self.discretise(observation)
        terminated = False
        truncated = False
        reward_sum = 0.0
        
        while not (terminated or truncated):
            action = self.pick_action(observation)
            new_observation, reward, terminated, truncated, _ = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        # Create bins for each observation dimension
        bins = []
        for i, (lower, upper, obs) in enumerate(zip(self.lower_bounds, 
                                                  self.upper_bounds, 
                                                  observation)):
            bin_size = (upper - lower) / self.n_bins
            bin_number = int((obs - lower) / bin_size)
            # Clip to ensure we stay within valid bins
            bin_number = max(0, min(self.n_bins - 1, bin_number))
            bins.append(bin_number)
        return tuple(bins)

    def pick_action(self, observation):
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return self.environment.action_space.sample()
        
        # Get Q-values for this state
        state_key = observation
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]
        
        return np.argmax(self.q_table[state_key])

    def update_knowledge(self, action, observation, new_observation, reward):
        # Get current and next state keys
        current_state = observation
        next_state = new_observation
        
        # Initialize Q-values if not exists
        if current_state not in self.q_table:
            self.q_table[current_state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        # Q-learning update formula
        current_q = self.q_table[current_state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[current_state][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)


def main():
    learner = QLearner()
    learner.learn(10000)


if __name__ == '__main__':
    main()
