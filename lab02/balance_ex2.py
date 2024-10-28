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

class QLearningExperiment:
    """Handles multiple runs of Q-Learning experiments and their analysis."""
    
    def __init__(self, num_runs: int = 5, max_episodes: int = 1000, params: ParameterSet = None):
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
        self.all_runs_data = []
        
    def run_experiments(self):
        """Run multiple Q-learning experiments and collect data."""
        all_rewards = []
        all_epsilons = []
        q_table_sizes = []
        
        for run in range(self.num_runs):
            print(f"\nStarting Run {run + 1}/{self.num_runs} for {self.params.name}")
            learner = QLearner(
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
            'params': self.params.to_dict()
        }
    
    def _save_run_data(self, run_data: dict, run_number: int):
        """Save individual run data to file."""
        run_file = self.results_dir / f'run_{run_number}.json'
        with open(run_file, 'w') as f:
            json.dump(run_data, f)
            
    def _analyze_results(self):
        """Analyze results across all runs and generate visualizations."""
        # Combine all runs into a DataFrame
        all_rewards = pd.DataFrame()
        for run_idx, run_data in enumerate(self.all_runs_data):
            rewards = pd.DataFrame({
                'episode': range(len(run_data['rewards'])),
                'reward': run_data['rewards'],
                'run': run_idx
            })
            all_rewards = pd.concat([all_rewards, rewards])
        
        # Calculate statistics
        stats = (all_rewards.groupby('episode')
                .agg({'reward': ['mean', 'std']})
                .reward)
        
        # Generate plots
        self._plot_learning_curves(stats)
        self._plot_run_comparisons(all_rewards)
        
    def _plot_learning_curves(self, stats):
        """Plot learning curves with confidence intervals."""
        plt.figure(figsize=(12, 6))
        window_size = 50  # Moving average window
        
        # Plot mean with confidence interval
        mean_smooth = stats['mean'].rolling(window=window_size).mean()
        std_smooth = stats['std'].rolling(window=window_size).mean()
        
        plt.plot(mean_smooth, 'b-', label='Moving Average (50 episodes)')
        plt.fill_between(
            range(len(mean_smooth)),
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            alpha=0.2,
            color='b',
            label='±1 Standard Deviation'
        )
        
        plt.title('Learning Curve Across All Runs')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'learning_curves.png')
        plt.close()
        
    def _plot_run_comparisons(self, all_rewards):
        """Plot comparison of individual runs."""
        plt.figure(figsize=(12, 6))
        window_size = 50
        
        # Plot individual runs
        for run in range(self.num_runs):
            run_data = all_rewards[all_rewards['run'] == run]
            smooth_rewards = run_data['reward'].rolling(window=window_size).mean()
            plt.plot(smooth_rewards, alpha=0.5, label=f'Run {run + 1}')
            
        plt.title('Individual Run Comparisons')
        plt.xlabel('Episode')
        plt.ylabel('Smoothed Reward (50-episode window)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'run_comparisons.png')
        plt.close()


class QLearner:
    """Q-Learning implementation for CartPole environment."""
    
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
        
        # Q-learning parameters
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
        
    def learn(self, max_episodes: int) -> dict:
        """Run the learning process and return collected data."""
        for episode in range(max_episodes):
            reward_sum = self._run_episode()
            self.rewards_history.append(reward_sum)
            self.epsilon_history.append(self.epsilon)
            
            # Optional progress printing
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
        """Run a single episode and return the total reward."""
        observation, _ = self.environment.reset()
        observation = self._discretize_state(observation)
        terminated = False
        truncated = False
        total_reward = 0.0
        
        while not (terminated or truncated):
            action = self._select_action(observation)
            new_observation, reward, terminated, truncated, _ = self.environment.step(action)
            new_observation = self._discretize_state(new_observation)
            
            # Apply reward shaping
            angle_penalty = abs(new_observation[2]) * 0.1
            modified_reward = reward - angle_penalty
            
            self._update_q_table(action, observation, new_observation, modified_reward)
            observation = new_observation
            total_reward += reward
            
        return total_reward
    
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
    
    def _update_q_table(self, action: int, state: Tuple, 
                       next_state: Tuple, reward: float) -> None:
        """Update Q-table using Q-learning update rule."""
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
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_min, 
                         self.epsilon * self.epsilon_decay)


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create and run experiment
    experiment = QLearningExperiment(num_runs=5, max_episodes=1000)
    experiment.run_experiments()


if __name__ == '__main__':
    main()