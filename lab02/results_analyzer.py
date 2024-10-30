from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsAnalyzer:
    """Universal analyzer for all types of experiment results."""
    
    def __init__(self, timestamp: str, experiment_type: str = 'algorithm_comparison', window_size: int = 1000):
        """
        Initialize the analyzer.
        
        Args:
            timestamp: Timestamp of the experiment
            experiment_type: Type of experiment ('algorithm_comparison', 'parameter_study', or 'single_run')
            window_size: Window size for moving average calculations
        """
        self.experiment_type = experiment_type
        self.results_dir = Path(experiment_type) / timestamp
        self.window_size = window_size
        
        if not self.results_dir.exists():
            raise ValueError(f"No results found for timestamp: {timestamp} in {experiment_type}")
        
        self.all_results = {}
        self._load_results()
    
    def _load_results(self):
        """Load results based on experiment type."""
        if self.experiment_type == 'single_run':
            self._load_single_run_results()
        else:
            self._load_multiple_results()
    
    def _load_single_run_results(self):
        """Load results from a single run experiment."""
        results_file = self.results_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            # Store under the algorithm name for consistent processing
            with open(self.results_dir / 'parameters.json', 'r') as f:
                params = json.load(f)
            key = f"{results['algorithm']}_{params['name']}"
            self.all_results[key] = results
    
    def _load_multiple_results(self):
        """Load results from comparison or parameter study experiments."""
        for result_dir in self.results_dir.iterdir():
            if result_dir.is_dir():
                results_file = result_dir / 'results.json'
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        self.all_results[result_dir.name] = json.load(f)
    
    def reanalyze(self):
        """Regenerate analysis plots with current settings."""
        self._plot_learning_curves()
        self._plot_performance_boxplots()
        self._generate_summary_statistics()
        if self.experiment_type != 'single_run':
            self._plot_parameter_heatmap()
    
    def _plot_learning_curves(self):
        """Plot learning curves for all variants."""
        plt.figure(figsize=(15, 8))
        
        for name, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            if rewards.ndim == 1:  # Single run case
                rewards = rewards.reshape(1, -1)
            
            mean_rewards = rewards.mean(axis=0)
            std_rewards = rewards.std(axis=0)
            
            # Calculate moving averages
            mean_smooth = pd.Series(mean_rewards).rolling(window=self.window_size).mean()
            std_smooth = pd.Series(std_rewards).rolling(window=self.window_size).mean()
            
            plt.plot(mean_smooth, label=name)
            plt.fill_between(
                range(len(mean_smooth)),
                mean_smooth - std_smooth,
                mean_smooth + std_smooth,
                alpha=0.2
            )
        
        plt.title(f'Learning Curves Comparison (Window Size: {self.window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'learning_curves_window_{self.window_size}.png')
        plt.close()
    
    def _plot_performance_boxplots(self):
        """Create boxplots comparing final performance."""
        final_rewards = []
        names = []
        
        for name, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            if rewards.ndim == 1:  # Single run case
                rewards = rewards.reshape(1, -1)
            
            final_100_rewards = rewards[:, -100:].mean(axis=1)
            final_rewards.extend(final_100_rewards)
            names.extend([name] * len(final_100_rewards))
        
        df = pd.DataFrame({
            'Variant': names,
            'Average Reward (Final 100 Episodes)': final_rewards
        })
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Variant', y='Average Reward (Final 100 Episodes)')
        plt.title('Final Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / f'performance_boxplot_window_{self.window_size}.png')
        plt.close()
    
    def _plot_parameter_heatmap(self):
        """Create heatmap of parameter effects (for parameter study only)."""
        if self.experiment_type != 'parameter_study':
            return
            
        param_data = []
        for name, results in self.all_results.items():
            params = results['params']
            rewards = np.array(results['all_rewards'])
            final_performance = rewards[:, -100:].mean()
            param_data.append({**params, 'performance': final_performance})
        
        if param_data:
            df = pd.DataFrame(param_data)
            
            # Create correlation matrix
            correlation_matrix = df.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Parameter Correlation with Performance')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'parameter_correlation_heatmap.png')
            plt.close()
    
    def _generate_summary_statistics(self):
        """Generate and save summary statistics."""
        summary = {}
        
        for name, results in self.all_results.items():
            rewards = np.array(results['all_rewards'])
            if rewards.ndim == 1:  # Single run case
                rewards = rewards.reshape(1, -1)
            
            final_100_rewards = rewards[:, -100:].mean(axis=1)
            
            summary[name] = {
                'mean_final_reward': float(final_100_rewards.mean()),
                'std_final_reward': float(final_100_rewards.std()),
                'max_reward': float(rewards.max()),
                'min_reward': float(rewards.min()),
                'mean_convergence_episode': float(
                    np.argmax(rewards.mean(axis=0) > 195)
                    if any(rewards.mean(axis=0) > 195)
                    else len(rewards[0])
                ),
                'q_table_size': results.get('q_table_sizes', [0])[0]
            }
            
            # Add parameters if available
            if 'params' in results:
                summary[name]['parameters'] = results['params']
        
        with open(self.results_dir / f'summary_statistics_window_{self.window_size}.json', 'w') as f:
            json.dump(summary, f, indent=4)
    