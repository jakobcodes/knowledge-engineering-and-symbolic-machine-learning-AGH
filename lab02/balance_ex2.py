import math
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns
from pathlib import Path
import json
from parameter_set import ParameterSet

class BaseLearner:
    """Base class for reinforcement learning algorithms."""
    
    def __init__(self, learning_rate: float, discount_factor: float,
                 epsilon_start: float, epsilon_decay: float,
                 epsilon_min: float, n_bins: int):
        self.environment = gym.make('CartPole-v1')
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
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_bins = n_bins
        self.q_table: Dict[Tuple, List[float]] = {}
        
        # Performance tracking
        self.rewards_history = []
        self.epsilon_history = []
    
    def _discretize_state(self, observation: np.ndarray) -> Tuple:
        """Convert continuous state to discrete state."""
        bins = []
        for i, (lower, upper, obs) in enumerate(zip(self.lower_bounds, 
                                                  self.upper_bounds, 
                                                  observation)):
            bin_size = (upper - lower) / self.n_bins
            bin_number = int((obs - lower) / bin_size)
            bin_number = max(0, min(self.n_bins - 1, bin_number))
            bins.append(bin_number)
        return tuple(bins)
    
    def _select_action(self, state: Tuple) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.environment.action_space.sample()
        
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        
        return np.argmax(self.q_table[state])

class QLearner(BaseLearner):
    """Q-Learning implementation."""
    
    def learn(self, max_episodes: int) -> dict:
        """Run the Q-learning process and return collected data."""
        for episode in range(max_episodes):
            reward_sum = self._run_episode()
            self.rewards_history.append(reward_sum)
            self.epsilon_history.append(self.epsilon)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return {
            'rewards': self.rewards_history,
            'epsilons': self.epsilon_history,
            'q_table_size': len(self.q_table)
        }
    
    def _run_episode(self) -> float:
        """Run a single Q-learning episode."""
        observation, _ = self.environment.reset()
        state = self._discretize_state(observation)
        terminated = False
        truncated = False
        total_reward = 0.0
        
        while not (terminated or truncated):
            action = self._select_action(state)
            new_observation, reward, terminated, truncated, _ = self.environment.step(action)
            new_state = self._discretize_state(new_observation)
            
            # Apply reward shaping
            angle_penalty = abs(new_state[2]) * 0.1
            modified_reward = reward - angle_penalty
            
            self._update_q_value(state, action, new_state, modified_reward)
            state = new_state
            total_reward += reward
            
            # Update exploration rate
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon * self.epsilon_decay)
        
        return total_reward
    
    def _update_q_value(self, state: Tuple, action: int, 
                       next_state: Tuple, reward: float) -> None:
        """Update Q-value using Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state][action] = new_q

class SARSALearner(BaseLearner):
    """SARSA implementation."""
    
    def learn(self, max_episodes: int) -> dict:
        """Run the SARSA learning process and return collected data."""
        for episode in range(max_episodes):
            reward_sum = self._run_episode()
            self.rewards_history.append(reward_sum)
            self.epsilon_history.append(self.epsilon)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.rewards_history[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return {
            'rewards': self.rewards_history,
            'epsilons': self.epsilon_history,
            'q_table_size': len(self.q_table)
        }
    
    def _run_episode(self) -> float:
        """Run a single SARSA episode."""
        observation, _ = self.environment.reset()
        state = self._discretize_state(observation)
        action = self._select_action(state)  # Select first action
        terminated = False
        truncated = False
        total_reward = 0.0
        
        while not (terminated or truncated):
            new_observation, reward, terminated, truncated, _ = self.environment.step(action)
            new_state = self._discretize_state(new_observation)
            new_action = self._select_action(new_state)  # Select next action
            
            # Apply reward shaping
            angle_penalty = abs(new_state[2]) * 0.1
            modified_reward = reward - angle_penalty
            
            self._update_q_value(state, action, new_state, new_action, modified_reward)
            
            state = new_state
            action = new_action
            total_reward += reward
            
            # Update exploration rate
            self.epsilon = max(self.epsilon_min, 
                             self.epsilon * self.epsilon_decay)
        
        return total_reward
    
    def _update_q_value(self, state: Tuple, action: int, 
                       next_state: Tuple, next_action: int, reward: float) -> None:
        """Update Q-value using SARSA update rule."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0, 0.0]
        
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        self.q_table[state][action] = new_q

# Update QLearningExperiment to handle both algorithms
class ReinforcementLearningExperiment:
    """Handles multiple runs of reinforcement learning experiments and their analysis."""
    
    def __init__(self, algorithm: str = "q_learning", num_runs: int = 5, 
                 max_episodes: int = 1000, params: ParameterSet = None):
        self.algorithm = algorithm
        self.num_runs = num_runs
        self.max_episodes = max_episodes
        self.params = params or ParameterSet(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            n_bins=6,
            name="Default"
        )
    
    def run_experiments(self):
        """Run multiple learning experiments and collect data."""
        all_rewards = []
        all_epsilons = []
        q_table_sizes = []
        
        for run in range(self.num_runs):
            print(f"\nStarting Run {run + 1}/{self.num_runs} for {self.params.name}")
            
            # Create appropriate learner based on algorithm
            if self.algorithm == "q_learning":
                learner = QLearner(
                    learning_rate=self.params.learning_rate,
                    discount_factor=self.params.discount_factor,
                    epsilon_start=self.params.epsilon_start,
                    epsilon_decay=self.params.epsilon_decay,
                    epsilon_min=self.params.epsilon_min,
                    n_bins=self.params.n_bins
                )
            else:  # SARSA
                learner = SARSALearner(
                    learning_rate=self.params.learning_rate,
                    discount_factor=self.params.discount_factor,
                    epsilon_start=self.params.epsilon_start,
                    epsilon_decay=self.params.epsilon_decay,
                    epsilon_min=self.params.epsilon_min,
                    n_bins=self.params.n_bins
                )
            
            run_data = learner.learn(self.max_episodes)
            all_rewards.append(run_data['rewards'])
            all_epsilons.append(run_data['epsilons'])
            q_table_sizes.append(run_data['q_table_size'])
        
        return {
            'all_rewards': all_rewards,
            'all_epsilons': all_epsilons,
            'q_table_sizes': q_table_sizes,
            'params': self.params.to_dict(),
            'algorithm': self.algorithm
        }


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run algorithm comparison
    comparison = AlgorithmComparison(num_runs=5, max_episodes=1000)
    comparison.run_comparison()

if __name__ == '__main__':
    main()