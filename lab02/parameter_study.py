import math
import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import seaborn as sns

from itertools import product
from balance_ex2 import QLearningExperiment
from parameter_set import ParameterSet


class ParameterStudy:
    """Manages multiple Q-learning experiments with different parameter sets."""
    
    def __init__(self, num_runs: int = 5, max_episodes: int = 1000):
        self.num_runs = num_runs
        self.max_episodes = max_episodes
        self.results_dir = Path('parameter_study') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define parameter sets for study
        self.parameter_sets = [
            ParameterSet(
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon_start=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                n_bins=6,
                name="Baseline"
            ),
            ParameterSet(
                learning_rate=0.01,
                discount_factor=0.99,
                epsilon_start=1.0,
                epsilon_decay=0.99,
                epsilon_min=0.05,
                n_bins=8,
                name="Conservative"
            ),
            ParameterSet(
                learning_rate=0.3,
                discount_factor=0.9,
                epsilon_start=1.0,
                epsilon_decay=0.95,
                epsilon_min=0.01,
                n_bins=4,
                name="Aggressive"
            ),
            ParameterSet(
                learning_rate=0.1,
                discount_factor=0.95,
                epsilon_start=0.5,
                epsilon_decay=0.999,
                epsilon_min=0.1,
                n_bins=10,
                name="Fine-Grained"
            ),
        ]
        
        self.all_results = {}
    
    def run_study(self):
        """Run experiments for all parameter sets."""
        for param_set in self.parameter_sets:
            print(f"\nTesting parameter set: {param_set.name}")
            experiment = QLearningExperiment(
                num_runs=self.num_runs,
                max_episodes=self.max_episodes,
                params=param_set
            )
            results = experiment.run_experiments()
            self.all_results[param_set.name] = results
            
            # Save parameter set results
            self._save_parameter_results(param_set, results)
        
        # Generate comparative analysis
        self._analyze_all_results()
    
    def _save_parameter_results(self, param_set: ParameterSet, results: dict):
        """Save results for a parameter set."""
        param_dir = self.results_dir / param_set.name
        param_dir.mkdir(exist_ok=True)
        
        # Save parameters
        with open(param_dir / 'parameters.json', 'w') as f:
            json.dump(param_set.to_dict(), f, indent=4)
        
        # Save results
        with open(param_dir / 'results.json', 'w') as f:
            json.dump(results, f)
    
    def _analyze_all_results(self):
        """Generate comparative analysis of all parameter sets."""
        self._plot_comparative_learning_curves()
        self._plot_parameter_boxplots()
        self._generate_summary_statistics()
    
    def _plot_comparative_learning_curves(self):
        """Plot learning curves for all parameter sets."""
        plt.figure(figsize=(15, 8))
        window_size = 50
        
        for param_name, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            mean_rewards = rewards.mean(axis=0)
            std_rewards = rewards.std(axis=0)
            
            # Calculate moving averages
            mean_smooth = pd.Series(mean_rewards).rolling(window=window_size).mean()
            std_smooth = pd.Series(std_rewards).rolling(window=window_size).mean()
            
            plt.plot(mean_smooth, label=param_name)
            plt.fill_between(
                range(len(mean_smooth)),
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                alpha=0.2
            )
        
        plt.title('Learning Curves Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'comparative_learning_curves.png')
        plt.close()
    
    def _plot_parameter_boxplots(self):
        """Create boxplots comparing final performance across parameter sets."""
        final_rewards = []
        param_names = []
        
        # Collect final 100 episodes for each run of each parameter set
        for param_name, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            final_100_rewards = rewards[:, -100:].mean(axis=1)
            final_rewards.extend(final_100_rewards)
            param_names.extend([param_name] * len(final_100_rewards))
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Parameter Set': param_names,
            'Average Reward (Final 100 Episodes)': final_rewards
        })
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Parameter Set', y='Average Reward (Final 100 Episodes)')
        plt.title('Final Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'parameter_comparison_boxplot.png')
        plt.close()
    
    def _generate_summary_statistics(self):
        """Generate and save summary statistics for all parameter sets."""
        summary = {}
        
        for param_name, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            final_100_rewards = rewards[:, -100:].mean(axis=1)
            
            summary[param_name] = {
                'mean_final_reward': float(final_100_rewards.mean()),
                'std_final_reward': float(final_100_rewards.std()),
                'max_reward': float(rewards.max()),
                'mean_convergence_episode': float(
                    np.argmax(rewards.mean(axis=0) > 195)
                    if any(rewards.mean(axis=0) > 195)
                    else self.max_episodes
                )
            }
        
        with open(self.results_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=4)

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create and run parameter study
    study = ParameterStudy(num_runs=5, max_episodes=1000)
    study.run_study()

if __name__ == '__main__':
    main()