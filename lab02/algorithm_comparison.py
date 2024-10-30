from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from parameter_set import ParameterSet
from balance_ex2 import ReinforcementLearningExperiment

class AlgorithmComparison:
    """Compares Q-Learning and SARSA algorithms."""
    
    def __init__(self, num_runs: int = 5, max_episodes: int = 1000):
        self.num_runs = num_runs
        self.max_episodes = max_episodes
        self.results_dir = Path('algorithm_comparison') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a single parameter set for comparison
        self.params = ParameterSet(
            learning_rate=0.3,
            discount_factor=0.9,
            epsilon_start=1.0,
            epsilon_decay=0.95,
            epsilon_min=0.01,
            n_bins=4,
            name="Default"
        )
        
        self.algorithms = ['q_learning', 'sarsa']
        self.all_results = {}

    def run_comparison(self):
        """Run experiments for both algorithms."""
        for algorithm in self.algorithms:
            print(f"\nTesting algorithm: {algorithm}")
            experiment = ReinforcementLearningExperiment(
                algorithm=algorithm,
                num_runs=self.num_runs,
                max_episodes=self.max_episodes,
                params=self.params
            )
            results = experiment.run_experiments()
            self.all_results[algorithm] = results

            # Save results
            self._save_results(algorithm, results)

        # Generate comparative analysis
        self._analyze_results()

    def _save_results(self, algorithm: str, results: dict):
        """Save results for an algorithm."""
        algo_dir = self.results_dir / algorithm
        algo_dir.mkdir(exist_ok=True)

        # Save parameters and results
        with open(algo_dir / 'parameters.json', 'w') as f:
            json.dump(self.params.to_dict(), f, indent=4)

        with open(algo_dir / 'results.json', 'w') as f:
            json.dump(results, f)

    def _analyze_results(self):
        """Generate comparative analysis of both algorithms."""
        self._plot_learning_curves()
        self._plot_algorithm_boxplots()
        self._generate_summary_statistics()

    def _plot_learning_curves(self):
        """Plot learning curves comparing both algorithms."""
        plt.figure(figsize=(15, 8))
        window_size = 50

        for algorithm, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            mean_rewards = rewards.mean(axis=0)
            std_rewards = rewards.std(axis=0)

            # Calculate moving averages
            mean_smooth = pd.Series(mean_rewards).rolling(window=window_size).mean()
            std_smooth = pd.Series(std_rewards).rolling(window=window_size).mean()

            plt.plot(mean_smooth, label=algorithm.upper())
            plt.fill_between(
                range(len(mean_smooth)),
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                alpha=0.2
            )

        plt.title('Algorithm Learning Curves Comparison')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.results_dir / 'learning_curves_comparison.png')
        plt.close()

    def _plot_algorithm_boxplots(self):
        """Create boxplots comparing final performance of both algorithms."""
        final_rewards = []
        algorithm_names = []

        for algorithm, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            final_100_rewards = rewards[:, -100:].mean(axis=1)
            final_rewards.extend(final_100_rewards)
            algorithm_names.extend([algorithm.upper()] * len(final_100_rewards))

        df = pd.DataFrame({
            'Algorithm': algorithm_names,
            'Average Reward (Final 100 Episodes)': final_rewards
        })

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Algorithm', y='Average Reward (Final 100 Episodes)')
        plt.title('Final Performance Comparison')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'algorithm_comparison_boxplot.png')
        plt.close()

    def _generate_summary_statistics(self):
        """Generate and save summary statistics for both algorithms."""
        summary = {}

        for algorithm, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            final_100_rewards = rewards[:, -100:].mean(axis=1)

            summary[algorithm] = {
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