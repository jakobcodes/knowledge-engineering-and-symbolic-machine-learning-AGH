import math
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict
import matplotlib.pyplot as plt

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
        
        # Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.n_bins = 6
        self.q_table: Dict[Tuple, list] = {}
        
        # Performance tracking
        self.rewards_history = []
        self.epsilon_history = []

    def learn(self, max_attempts: int) -> None:
        for episode in range(max_attempts):
            reward_sum = self.attempt()
            self.rewards_history.append(reward_sum)
            self.epsilon_history.append(self.epsilon)
            
            # Print progress every 100 episodes
            if episode % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
            
            # Early stopping if solved (avg reward > 195 over 100 episodes)
            if len(self.rewards_history) >= 100 and np.mean(self.rewards_history[-100:]) > 195:
                print(f"Environment solved in {episode} episodes!")
                break

    def attempt(self) -> float:
        observation, _ = self.environment.reset()
        observation = self.discretise(observation)
        terminated = False
        truncated = False
        reward_sum = 0.0
        
        while not (terminated or truncated):
            action = self.pick_action(observation)
            new_observation, reward, terminated, truncated, _ = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            
            # Modify reward to encourage pole staying upright
            angle = new_observation[2]  # Index for pole angle
            modified_reward = reward - abs(angle) * 0.1  # Penalize large angles
            
            self.update_knowledge(action, observation, new_observation, modified_reward)
            observation = new_observation
            reward_sum += reward
        
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation: np.ndarray) -> Tuple:
        bins = []
        for i, (lower, upper, obs) in enumerate(zip(self.lower_bounds, 
                                                  self.upper_bounds, 
                                                  observation)):
            bin_size = (upper - lower) / self.n_bins
            bin_number = int((obs - lower) / bin_size)
            bin_number = max(0, min(self.n_bins - 1, bin_number))
            bins.append(bin_number)
        return tuple(bins)

    def pick_action(self, observation: Tuple) -> int:
        if np.random.random() < self.epsilon:
            return self.environment.action_space.sample()
        
        state_key = observation
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0]
        
        return np.argmax(self.q_table[state_key])

    def update_knowledge(self, action: int, observation: Tuple, 
                        new_observation: Tuple, reward: float) -> None:
        current_state = observation
        next_state = new_observation
        
        if current_state not in self.q_table:
            self.q_table[current_state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        current_q = self.q_table[current_state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[current_state][action] = new_q
        
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)

    def plot_performance(self) -> None:
        """Plot the learning progress."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot rewards
        ax1.plot(self.rewards_history)
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        
        # Plot moving average
        window_size = 10
        moving_avg = np.convolve(self.rewards_history, 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        ax1.plot(moving_avg, 'r-', label=f'{window_size}-episode moving average')
        ax1.legend()
        
        # Plot epsilon decay
        ax2.plot(self.epsilon_history)
        ax2.set_title('Epsilon Decay')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()


def main():
    learner = QLearner()
    learner.learn(200)
    learner.plot_performance()


if __name__ == '__main__':
    main()