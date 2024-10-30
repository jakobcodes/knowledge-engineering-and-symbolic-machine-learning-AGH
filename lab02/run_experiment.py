import numpy as np
from parameter_study import ParameterStudy
from algorithm_comparison import AlgorithmComparison
from results_analyzer import ResultsAnalyzer
import argparse
from single_run import SingleRun
from parameter_set import ParameterSet
import json

def run_algorithm_comparison(num_runs: int, max_episodes: int):
    """Run comparison between SARSA and Q-Learning"""
    comparison = AlgorithmComparison(num_runs=num_runs, max_episodes=max_episodes)
    comparison.run_comparison()

def run_parameter_study(algorithm: str, num_runs: int, max_episodes: int):
    """Run learning using single algorithm with different parameters"""
    study = ParameterStudy(algorithm=algorithm, num_runs=num_runs, max_episodes=max_episodes)
    study.run_study()

def analyze_results(timestamp: str, experiment_type: str, window_size: int):
    """Generate plots with different window size for existing experiments"""
    analyzer = ResultsAnalyzer(
        timestamp=timestamp,
        experiment_type=experiment_type,
        window_size=window_size
    )
    analyzer.reanalyze()
    print(f"\nAnalysis completed for {experiment_type}")
    print(f"Results saved in: {analyzer.results_dir}")

def run_single(algorithm: str, max_episodes: int, params: dict = None):
    """Run a single learning process with specified parameters and analyze results"""
    if params:
        parameter_set = ParameterSet(**params)
    else:
        parameter_set = None
        
    # Run the experiment
    runner = SingleRun(algorithm=algorithm, max_episodes=max_episodes, params=parameter_set)
    results = runner.run()
    timestamp = runner.results_dir.name  # Get the timestamp from the results directory
    
    print(f"\nResults saved in: {runner.results_dir}")
    print("\nInitial visualization files generated:")
    print("- learning_curve.png")
    print("- epsilon_decay.png")
    print("- reward_distribution.png")
    print("- summary_statistics.json")
    
    # Automatically analyze results with different window sizes
    window_sizes = [50, 100, 200, 500, 1000]  # Multiple window sizes for different granularity
    print("\nGenerating additional analysis with different window sizes...")
    
    for window_size in window_sizes:
        analyze_results(timestamp, "single_run", window_size)
        print(f"- Analysis with window size {window_size} completed")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    parser.add_argument('mode', choices=['compare', 'parameter_study', 'analyze', 'single'],
                       help='Mode of operation')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of runs for each experiment')
    parser.add_argument('--max_episodes', type=int, default=1000,
                       help='Maximum number of episodes per run')
    parser.add_argument('--timestamp', type=str,
                       help='Timestamp of experiment to analyze')
    parser.add_argument('--window_size', type=int, default=1000,
                       help='Window size for moving average in plots')
    parser.add_argument('--algorithm', type=str, choices=['q_learning', 'sarsa'],
                       help='Algorithm to use for single run')
    parser.add_argument('--params', type=str,
                       help='JSON string with parameters for single run')
    parser.add_argument('--experiment_type', 
                       choices=['algorithm_comparison', 'parameter_study', 'single_run'],
                       help='Type of experiment to analyze')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    if args.mode == 'compare':
        run_algorithm_comparison(args.num_runs, args.max_episodes)
    elif args.mode == 'parameter_study':
        run_parameter_study(args.algorithm, args.num_runs, args.max_episodes)
    elif args.mode == 'analyze':
        if not args.timestamp:
            raise ValueError("Timestamp is required for analysis mode")
        if not args.experiment_type:
            raise ValueError("Experiment type is required for analysis mode")
        analyze_results(args.timestamp, args.experiment_type, args.window_size)
    elif args.mode == 'single':
        if not args.algorithm:
            raise ValueError("Algorithm must be specified for single run")
        params = json.loads(args.params) if args.params else None
        run_single(args.algorithm, args.max_episodes, params)

if __name__ == '__main__':
    main() 