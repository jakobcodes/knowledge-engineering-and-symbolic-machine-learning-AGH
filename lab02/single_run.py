from balance_ex2 import ReinforcementLearningExperiment
from parameter_set import ParameterSet
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

class SingleRun:
    """Handles a single run of the reinforcement learning algorithm."""
    
    def __init__(self, algorithm: str, max_episodes: int, params: ParameterSet = None):
        self.algorithm = algorithm
        self.max_episodes = max_episodes
        self.params = params or ParameterSet()
        self.results_dir = Path('single_run') / datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def run(self):
        """Execute a single learning run and save results."""
        print(f"\nStarting single run with {self.algorithm}")
        print(f"Parameters: {self.params.name}")
        
        start_time = time.time()
        
        experiment = ReinforcementLearningExperiment(
            algorithm=self.algorithm,
            num_runs=1,
            max_episodes=self.max_episodes,
            params=self.params
        )
        
        results = experiment.run_experiments()
        
        training_time = time.time() - start_time
        results['training_time'] = training_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"                     = {training_time/60:.2f} minutes")
        print(f"                     = {training_time/3600:.2f} hours")
        print(f"Average time per episode: {(training_time/self.max_episodes)*1000:.2f} ms")
        
        self._save_results(results)
        self._generate_visualizations(results)
        
        return results
        
    def _save_results(self, results: dict):
        """Save the results and parameters to files."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save parameters
        with open(self.results_dir / 'parameters.json', 'w') as f:
            json.dump(self.params.to_dict(), f, indent=4)
        
        # Save results
        with open(self.results_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
    def _generate_visualizations(self, results: dict):
        """Generate and save visualization plots."""
        self._plot_learning_curve(results)
        self._plot_epsilon_decay(results)
        self._plot_reward_distribution(results)
        self._generate_summary_statistics(results)
        
    def _plot_learning_curve(self, results: dict):
        """Plot the learning curve with moving average."""
        plt.figure(figsize=(12, 6))
        rewards = results['all_rewards'][0]  # Single run, so take first element
        
        # Calculate moving average
        window_size = 50
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw rewards')
        plt.plot(moving_avg, color='red', label=f'Moving average (window={window_size})')
        
        plt.title(f'Learning Curve - {self.algorithm}')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'learning_curve.png')
        plt.close()
        
    def _plot_epsilon_decay(self, results: dict):
        """Plot the epsilon decay over episodes."""
        plt.figure(figsize=(10, 5))
        epsilons = results['all_epsilons'][0]  # Single run
        
        plt.plot(epsilons, color='green')
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'epsilon_decay.png')
        plt.close()
        
    def _plot_reward_distribution(self, results: dict):
        """Plot the distribution of rewards."""
        plt.figure(figsize=(10, 5))
        rewards = results['all_rewards'][0]
        
        # Create histogram and KDE plot
        sns.histplot(rewards, kde=True)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.savefig(self.results_dir / 'reward_distribution.png')
        plt.close()
        
    def _generate_summary_statistics(self, results: dict):
        """Generate and save summary statistics."""
        rewards = np.array(results['all_rewards'][0])
        
        summary = {
            'algorithm': self.algorithm,
            'statistics': {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'final_100_episodes_mean': float(np.mean(rewards[-100:])),
                'q_table_size': results['q_table_sizes'][0],
                'training_time_seconds': float(results['training_time']),
                'time_per_episode_ms': float((results['training_time']/self.max_episodes)*1000)
            },
            'parameters': self.params.to_dict()
        }
        
        with open(self.results_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=4) 