"""
Script to create plots from best response pkl.gz files.

Can plot single or multiple files to compare different algorithms.
"""

import os
import sys
import argparse
import gzip
import pickle
import math
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.best_response_evaluator import BestResponseTracker


def load_best_response_data(filepath, suppress_warnings=True):
    """
    Load best response data from a pkl.gz file.
    
    Args:
        filepath: Path to pkl.gz file
        suppress_warnings: If True, warnings are suppressed
    
    Returns:
        BestResponseTracker object or None on error
    """
    try:
        # Load directly from file, without calling BestResponseTracker.__init__
        # (to avoid Public State Tree warning)
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Create minimal tracker object without calling __init__
        # We use object.__new__ to bypass __init__
        tracker = object.__new__(BestResponseTracker)

        # Set all necessary attributes manually
        tracker.game_name = data.get('game_name', 'unknown')
        loaded_values = data.get('values', [])

        # Backward compatibility: Convert old formats to new format
        if loaded_values:
            if len(loaded_values[0]) == 2:
                # Old format: (iteration, exploitability)
                tracker.values = [(it, exp, None, None, 0.0) for it, exp in loaded_values]
            elif len(loaded_values[0]) == 4:
                # Old format: (iteration, exploitability_mb, br_value_p0, br_value_p1)
                tracker.values = [(it, exp, br0, br1, 0.0) for it, exp, br0, br1 in loaded_values]
            else:
                # New format with time
                tracker.values = loaded_values
        else:
            tracker.values = []

        tracker.total_br_time = 0.0
        tracker.last_eval_iteration = 0

        # Schedule Config
        if 'schedule_config' in data:
            tracker.schedule_config = data['schedule_config']
            tracker.schedule_type = tracker.schedule_config.get("type", "fixed")

            if tracker.schedule_type == "fixed":
                tracker.interval = tracker.schedule_config.get("interval", 100)
            elif tracker.schedule_type == "logarithmic":
                tracker.base_interval = tracker.schedule_config.get("base_interval", 10)
                tracker.target_iteration = tracker.schedule_config.get("target_iteration", 100)
                tracker.target_interval = tracker.schedule_config.get("target_interval", 100)
                tracker.unbounded = tracker.schedule_config.get("unbounded", False)
            elif tracker.schedule_type == "custom":
                tracker.custom_schedule = tracker.schedule_config.get("schedule", [])
        else:
            # Fallback for old files
            tracker.schedule_config = {"type": "fixed", "interval": 100}
            tracker.schedule_type = "fixed"
            tracker.interval = 100

        # Tree is not loaded (not needed for plots)
        tracker.tree = None

        return tracker
    except Exception as e:
        if not suppress_warnings:
            print(f"Error loading {filepath}: {e}")
        return None


def extract_algorithm_name(filepath):
    """
    Extract algorithm name from file path.
    
    Example: data/models/leduc/cfr_optimized/1000/leduc_1000_best_response.pkl.gz
    -> "cfr_optimized"
    """
    parts = filepath.split(os.sep)
    # Search for known algorithm names in path
    for part in parts:
        if any(alg in part for alg in ['cfr', 'mccfr', 'chance', 'external']):
            return part
    # Fallback: use filename
    filename = os.path.basename(filepath)
    return filename.replace('_best_response.pkl.gz', '').replace('.pkl.gz', '')


def load_model_data(filepath):
    """
    Load training_time and iteration_count from a model file.
    
    Args:
        filepath: Path to model pkl.gz file (not best response file)
    
    Returns:
        Tuple (training_time, iteration_count) or (None, None) on error
    """
    try:
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        training_time = data.get('training_time', None)
        iteration_count = data.get('iteration_count', None)
        return training_time, iteration_count
    except Exception as e:
        return None, None


def get_model_filepath(best_response_filepath):
    """
    Convert a best response file path to a model file path.
    
    Args:
        best_response_filepath: Path to best response file
    
    Returns:
        Path to model file
    """
    # Replace "_best_response.pkl.gz" with ".pkl.gz"
    return best_response_filepath.replace('_best_response.pkl.gz', '.pkl.gz')


def plot_manual_time_comparison(filepaths, output_path=None, title=None, log_log=True, custom_labels=None):
    """
    Plot manually selected best response files against time.
    
    This function is specifically designed for manual file selection and offers
    more control over labels and presentation.
    
    Args:
        filepaths: List of paths to pkl.gz files
        output_path: Optional path to save plot
        title: Optional title for plot
        log_log: If True, both axes are logarithmically scaled
        custom_labels: Optional dictionary {filepath: label} for custom labels
    
    Returns:
        True if successful, False on error
    """
    if not filepaths:
        print("Error: No files specified for plotting")
        return False

    # Load all data
    trackers = []
    labels = []
    model_times = []
    valid_filepaths = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        tracker = load_best_response_data(filepath)
        if tracker is not None:
            # Load model file to get training_time and iteration_count
            model_path = get_model_filepath(filepath)
            training_time, iteration_count = load_model_data(model_path)

            if training_time is None or iteration_count is None:
                print(
                    f"Warning: Could not load training_time/iteration_count from {model_path}, skipping {filepath}")
                continue

            trackers.append(tracker)
            model_times.append((training_time, iteration_count))
            valid_filepaths.append(filepath)

            # Use custom_label if available, otherwise extract algorithm name
            if custom_labels and filepath in custom_labels:
                labels.append(custom_labels[filepath])
            else:
                labels.append(extract_algorithm_name(filepath))
        else:
            print(f"Warning: Could not load data from {filepath}, skipping")

    if not trackers:
        print("Error: No valid data found for plotting")
        return False

    print(f"\nPlotting {len(trackers)} files:")
    for filepath, label in zip(valid_filepaths, labels):
        print(f"  - {label}: {filepath}")

    # Create plot
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    plt.figure(figsize=(14, 8))

    # Colors for different algorithms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Markers for different algorithms
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']

    for i, (tracker, label, (total_time, total_iterations)) in enumerate(zip(trackers, labels, model_times)):
        if not tracker.values:
            print(f"Warning: {label} has no values")
            continue

        # Extract data
        iterations = [x[0] for x in tracker.values]
        exploitability = [x[1] for x in tracker.values]

        # Check if time is already stored in values (new format)
        if len(tracker.values[0]) >= 5:
            # New format: time is already stored
            times = [x[4] for x in tracker.values]
        else:
            # Old format: calculate time linearly (fallback)
            if total_iterations > 0 and total_time > 0:
                time_per_iteration = total_time / total_iterations
            else:
                # Fallback: use max_iter from best response values
                max_iter = max(iterations) if iterations else 1
                time_per_iteration = total_time / max_iter if max_iter > 0 else 0

            # Convert iterations to cumulative time
            times = [time_per_iteration * iter for iter in iterations]

        # Normalize time relative to first measurement point (all start at 0)
        if times:
            first_time = times[0]
            times = [t - first_time for t in times]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(times, exploitability,
                 label=label,
                 marker=marker,
                 linewidth=2,
                 markersize=6,
                 color=color,
                 markevery=max(1, len(times) // 20))

    # Axis scaling
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
        xlabel = 'Time (seconds, log scale)'
        ylabel = 'Exploitability (mb/g, log scale)'
    else:
        xlabel = 'Time (seconds)'
        ylabel = 'Exploitability (mb/g)'

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    elif trackers:
        game_name = trackers[0].game_name
        plt.title(f'Exploitability vs Time Comparison ({game_name})', fontsize=14, fontweight='bold')

    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Set meaningful ticks with scientific notation (10^x)
    if trackers:
        all_times = []
        all_exploitability = []
        for tracker, (total_time, total_iterations) in zip(trackers, model_times):
            if tracker.values:
                iterations = [x[0] for x in tracker.values]
                exploitability = [x[1] for x in tracker.values if x[1] > 0]

                # Check if time is already stored
                if len(tracker.values[0]) >= 5:
                    # New format: use stored time
                    times = [x[4] for x in tracker.values if x[1] > 0]
                    # Normalize time relative to first measurement point
                    if times:
                        first_time = times[0]
                        times = [t - first_time for t in times]
                else:
                    # Old format: calculate linearly
                    if total_iterations > 0 and total_time > 0:
                        time_per_iteration = total_time / total_iterations
                        times = [time_per_iteration * iter for iter in iterations if
                                 any(x[0] == iter and x[1] > 0 for x in tracker.values)]
                        # Normalize time relative to first measurement point
                        if times:
                            first_time = times[0]
                            times = [t - first_time for t in times]
                    else:
                        times = []

                all_times.extend(times)
                all_exploitability.extend(exploitability)

        if all_times and all_exploitability:
            min_time = min(t for t in all_times if t > 0)
            max_time = max(all_times)
            min_exp = min(all_exploitability)
            max_exp = max(all_exploitability)

            if log_log:
                # Log-log scaling: use scientific notation
                from matplotlib.ticker import LogLocator, LogFormatter
                plt.gca().xaxis.set_major_locator(LogLocator(base=10, numticks=10))
                plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=10))
                plt.gca().xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
                plt.gca().yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))

    # Save plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {output_path}")
    else:
        plt.show()

    plt.close()
    return True


def plot_multiple_best_responses_by_time(filepaths, output_path=None, title=None, log_log=True):
    """
    Plot multiple best response files against time instead of iterations.
    
    Args:
        filepaths: List of paths to pkl.gz files
        output_path: Optional path to save plot
        title: Optional title for plot
        log_log: If True, both axes are logarithmically scaled
    """
    if not filepaths:
        print("No files specified for plotting")
        return

    # Load all data
    trackers = []
    labels = []
    model_times = []

    for filepath in filepaths:
        tracker = load_best_response_data(filepath)
        if tracker is not None:
            # Load model file to get training_time and iteration_count
            model_path = get_model_filepath(filepath)
            training_time, iteration_count = load_model_data(model_path)

            if training_time is None or iteration_count is None:
                print(
                    f"Warning: Could not load training_time/iteration_count from {model_path}, skipping {filepath}")
                continue

            trackers.append(tracker)
            model_times.append((training_time, iteration_count))
            # Extract algorithm name for label
            alg_name = extract_algorithm_name(filepath)
            labels.append(alg_name)
        else:
            print(f"Skipping {filepath} due to error")

    if not trackers:
        print("No valid data found for plotting")
        return

    # Create plot
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

    plt.figure(figsize=(14, 8))

    # Colors for different algorithms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Markers for different algorithms
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']

    for i, (tracker, label, (total_time, total_iterations)) in enumerate(zip(trackers, labels, model_times)):
        if not tracker.values:
            print(f"Warning: {label} has no values")
            continue

        # Extract data
        iterations = [x[0] for x in tracker.values]
        exploitability = [x[1] for x in tracker.values]

        # Check if time is already stored in values (new format)
        if len(tracker.values[0]) >= 5:
            # New format: time is already stored
            times = [x[4] for x in tracker.values]
        else:
            # Old format: calculate time linearly (fallback)
            if total_iterations > 0 and total_time > 0:
                time_per_iteration = total_time / total_iterations
            else:
                # Fallback: use max_iter from best response values
                max_iter = max(iterations) if iterations else 1
                time_per_iteration = total_time / max_iter if max_iter > 0 else 0

            # Convert iterations to cumulative time
            times = [time_per_iteration * iter for iter in iterations]

        # Normalize time relative to first measurement point (all start at 0)
        if times:
            first_time = times[0]
            times = [t - first_time for t in times]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(times, exploitability,
                 label=label,
                 marker=marker,
                 linewidth=2,
                 markersize=6,
                 color=color,
                 markevery=max(1, len(times) // 20))

    # Axis scaling
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
        xlabel = 'Time (seconds, log scale)'
        ylabel = 'Exploitability (mb/g, log scale)'
    else:
        xlabel = 'Time (seconds)'
        ylabel = 'Exploitability (mb/g)'

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    elif trackers:
        game_name = trackers[0].game_name
        plt.title(f'Exploitability vs Time Comparison ({game_name})', fontsize=14, fontweight='bold')

    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Set meaningful ticks with scientific notation (10^x)
    if trackers:
        all_times = []
        all_exploitability = []
        for tracker, (total_time, total_iterations) in zip(trackers, model_times):
            if tracker.values:
                iterations = [x[0] for x in tracker.values]
                exploitability = [x[1] for x in tracker.values if x[1] > 0]

                # Check if time is already stored
                if len(tracker.values[0]) >= 5:
                    # New format: use stored time
                    times = [x[4] for x in tracker.values if x[1] > 0]
                    # Normalize time relative to first measurement point
                    if times:
                        first_time = times[0]
                        times = [t - first_time for t in times]
                else:
                    # Old format: calculate linearly
                    if total_iterations > 0 and total_time > 0:
                        time_per_iteration = total_time / total_iterations
                        times = [time_per_iteration * iter for iter in iterations if
                                 any(x[0] == iter and x[1] > 0 for x in tracker.values)]
                        # Normalize time relative to first measurement point
                        if times:
                            first_time = times[0]
                            times = [t - first_time for t in times]
                    else:
                        times = []

                all_times.extend(times)
                all_exploitability.extend(exploitability)

        if all_times:
            max_time = max(all_times)
            if log_log:
                # X-axis ticks - use only powers of 10
                min_time = min([t for t in all_times if t > 0])
                min_power = int(math.floor(math.log10(min_time))) if min_time > 0 else -2
                max_power = int(math.ceil(math.log10(max_time))) if max_time > 0 else 2
                x_ticks = [10 ** i for i in range(min_power, max_power + 1)]
                x_ticks = [t for t in x_ticks if t <= max_time * 1.1]
                x_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in x_ticks]
                plt.xticks(x_ticks, x_tick_labels)

                # Y-axis ticks - use only powers of 10
                if all_exploitability:
                    min_exp = min(all_exploitability)
                    max_exp = max(all_exploitability)

                    min_power = int(math.floor(math.log10(min_exp))) if min_exp > 0 else -2
                    max_power = int(math.ceil(math.log10(max_exp))) if max_exp > 0 else 2
                    y_ticks = [10 ** i for i in range(min_power, max_power + 1)]
                    y_ticks = [t for t in y_ticks if min_exp * 0.5 <= t <= max_exp * 2]

                    if y_ticks:
                        y_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in y_ticks]
                        plt.yticks(y_ticks, y_tick_labels)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
    else:
        plt.show()


def plot_multiple_best_responses(filepaths, output_path=None, title=None, log_log=True):
    """
    Plot multiple best response files on one graph.
    
    Args:
        filepaths: List of paths to pkl.gz files
        output_path: Optional path to save plot
        title: Optional title for plot
        log_log: If True, both axes are logarithmically scaled
    """
    if not filepaths:
        print("No files specified for plotting")
        return

    # Load all data
    trackers = []
    labels = []

    for filepath in filepaths:
        tracker = load_best_response_data(filepath)
        if tracker is not None:
            trackers.append(tracker)
            # Extract algorithm name for label
            alg_name = extract_algorithm_name(filepath)
            labels.append(alg_name)
        else:
            print(f"Skipping {filepath} due to error")

    if not trackers:
        print("No valid data found for plotting")
        return

    # Create plot
    # Enable LaTeX rendering for scientific notation
    plt.rcParams['text.usetex'] = False  # Disable LaTeX (requires LaTeX installation)
    plt.rcParams['mathtext.default'] = 'regular'  # Use regular math text

    plt.figure(figsize=(14, 8))

    # Colors for different algorithms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Markers for different algorithms
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']

    for i, (tracker, label) in enumerate(zip(trackers, labels)):
        if not tracker.values:
            print(f"Warning: {label} has no values")
            continue

        iterations = [x[0] for x in tracker.values]
        exploitability = [x[1] for x in tracker.values]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(iterations, exploitability,
                 label=label,
                 marker=marker,
                 linewidth=2,
                 markersize=6,
                 color=color,
                 markevery=max(1, len(iterations) // 20))  # Show not all markers

    # Axis scaling
    if log_log:
        plt.xscale('log')
        plt.yscale('log')
        xlabel = 'Iterations (log scale)'
        ylabel = 'Exploitability (mb/g, log scale)'
    else:
        xlabel = 'Iterations'
        ylabel = 'Exploitability (mb/g)'

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    elif trackers:
        game_name = trackers[0].game_name
        plt.title(f'Exploitability Comparison ({game_name})', fontsize=14, fontweight='bold')

    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')

    # Set meaningful ticks with scientific notation (10^x)
    if trackers:
        all_iterations = []
        all_exploitability = []
        for tracker in trackers:
            if tracker.values:
                all_iterations.extend([x[0] for x in tracker.values])
                all_exploitability.extend([x[1] for x in tracker.values if x[1] > 0])

        if all_iterations:
            max_iter = max(all_iterations)
            if log_log:
                # X-axis ticks - use only powers of 10
                import math
                min_iter = min(all_iterations)
                min_power = int(math.floor(math.log10(min_iter))) if min_iter > 0 else 0
                max_power = int(math.ceil(math.log10(max_iter))) if max_iter > 0 else 0
                x_ticks = [10 ** i for i in range(min_power, max_power + 1)]
                x_ticks = [t for t in x_ticks if t <= max_iter * 1.1]
                x_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in x_ticks]
                plt.xticks(x_ticks, x_tick_labels)

                # Y-axis ticks - use only powers of 10
                if all_exploitability:
                    min_exp = min(all_exploitability)
                    max_exp = max(all_exploitability)

                    min_power = int(math.floor(math.log10(min_exp))) if min_exp > 0 else -2
                    max_power = int(math.ceil(math.log10(max_exp))) if max_exp > 0 else 2
                    y_ticks = [10 ** i for i in range(min_power, max_power + 1)]
                    y_ticks = [t for t in y_ticks if min_exp * 0.5 <= t <= max_exp * 2]

                    if y_ticks:
                        y_tick_labels = [f'$10^{{{int(math.log10(t))}}}$' for t in y_ticks]
                        plt.yticks(y_ticks, y_tick_labels)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {output_path}")
    else:
        plt.show()


def is_sampling_algorithm(filepath):
    """
    Check if an algorithm is a sampling algorithm.
    
    Args:
        filepath: Path to best response file
    
    Returns:
        True if sampling algorithm, else False
    """
    sampling_keywords = ['chance_sampling', 'external_sampling', 'mccfr']
    return any(keyword in filepath for keyword in sampling_keywords)


def find_best_response_files(game, iterations, include_sampling=False):
    """
    Automatically find all best response files for a game and iteration count.
    
    Args:
        game: Name of the game (e.g. 'leduc')
        iterations: Number of iterations (e.g. 1000)
        include_sampling: If True, sampling algorithms are also included
    
    Returns:
        List of file paths
    """
    base_dir = 'data/models'
    game_dir = os.path.join(base_dir, game)

    if not os.path.exists(game_dir):
        print(f"Warning: Directory not found: {game_dir}")
        return []

    found_files = []
    iterations_str = str(iterations)

    # Search all subdirectories for matching files
    for root, dirs, files in os.walk(game_dir):
        # Check if directory contains exactly the desired iteration count
        # Use os.path.basename to check only the directory name
        dir_name = os.path.basename(root)
        if dir_name == iterations_str:
            for file in files:
                if file.endswith('_best_response.pkl.gz'):
                    # Check if filename contains exactly the iteration count
                    # e.g. "leduc_1000_best_response.pkl.gz" should match, but not "leduc_10000"
                    if f"_{iterations_str}_" in file or file.endswith(f"_{iterations_str}_best_response.pkl.gz"):
                        filepath = os.path.join(root, file)
                        # Filter sampling algorithms if not desired
                        if include_sampling or not is_sampling_algorithm(filepath):
                            found_files.append(filepath)

    return sorted(found_files)


def main():
    parser = argparse.ArgumentParser(
        description='Create plots from best response pkl.gz files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Creates BOTH plots automatically
  # 1. Non-sampling algorithms vs iterations
  # 2. All algorithms vs time
  uv run python src/plot_best_response.py --game leduc --iterations 1000

  # Plot only vs iterations (non-sampling)
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --iterations-only

  # Plot only vs time (all algorithms)
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --time-only

  # Plot only sampling algorithms vs time
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --sampling

  # With manual files (legacy mode)
  uv run python src/plot_best_response.py file1.pkl.gz file2.pkl.gz

  # Manual files vs time
  uv run python src/plot_best_response.py --time file1.pkl.gz file2.pkl.gz

  # Without log-log
  uv run python src/plot_best_response.py --game leduc --iterations 1000 --no-log-log
        """
    )

    parser.add_argument('files', nargs='*',
                        help='Paths to best response pkl.gz files (optional, if --game and --iterations are used)')
    parser.add_argument('--game', type=str, default=None,
                        help='Name of the game (e.g. leduc, kuhn_case1)')
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of iterations (e.g. 1000)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for plot (optional, automatically generated if --game and --iterations are used)')
    parser.add_argument('--title', type=str, default=None,
                        help='Title for plot')
    parser.add_argument('--no-log-log', action='store_true',
                        help='Disable log-log scaling (only X-axis logarithmic)')
    parser.add_argument('--sampling', action='store_true',
                        help='Plot only sampling algorithms vs time (overrides default behavior)')
    parser.add_argument('--iterations-only', action='store_true',
                        help='Plot only vs iterations, not vs time (overrides default behavior)')
    parser.add_argument('--time-only', action='store_true',
                        help='Plot only vs time, not vs iterations (overrides default behavior)')
    parser.add_argument('--time', action='store_true',
                        help='For manual files: plot vs time instead of vs iterations')

    args = parser.parse_args()

    # Determine which files to use
    if args.game and args.iterations:
        output_dir = 'data/plots/comparisons'
        os.makedirs(output_dir, exist_ok=True)

        # Default behavior: create both plots
        # 1. Non-sampling vs iterations
        # 2. All vs time

        if args.sampling:
            # Only sampling algorithms vs time
            print(f"Searching for sampling algorithms for {args.game} with {args.iterations} iterations...")
            valid_files = find_best_response_files(args.game, args.iterations, include_sampling=True)
            valid_files = [f for f in valid_files if is_sampling_algorithm(f)]

            if not valid_files:
                print(f"No sampling algorithms found for {args.game} with {args.iterations} iterations")
                return

            print(f"Found files ({len(valid_files)}):")
            for f in valid_files:
                print(f"  - {f}")

            output_path = args.output or os.path.join(output_dir,
                                                      f"{args.game}_{args.iterations}_sampling_time_comparison.png")
            title = args.title or f"Exploitability vs Time Comparison - Sampling Only ({args.game}, {args.iterations} iterations)"

            plot_multiple_best_responses_by_time(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )

        elif args.iterations_only:
            # Only non-sampling vs iterations
            print(f"Searching for non-sampling algorithms for {args.game} with {args.iterations} iterations...")
            valid_files = find_best_response_files(args.game, args.iterations, include_sampling=False)

            if not valid_files:
                print(f"No files found for {args.game} with {args.iterations} iterations")
                return

            print(f"Found files ({len(valid_files)}):")
            for f in valid_files:
                print(f"  - {f}")

            output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_comparison.png")
            title = args.title or f"Exploitability Comparison ({args.game}, {args.iterations} iterations)"

            plot_multiple_best_responses(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )

        elif args.time_only:
            # Only all vs time
            print(f"Searching for all algorithms for {args.game} with {args.iterations} iterations (time plot)...")
            valid_files = find_best_response_files(args.game, args.iterations, include_sampling=True)

            if not valid_files:
                print(f"No files found for {args.game} with {args.iterations} iterations")
                return

            print(f"Found files ({len(valid_files)}):")
            for f in valid_files:
                print(f"  - {f}")

            output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_time_comparison.png")
            title = args.title or f"Exploitability vs Time Comparison ({args.game}, {args.iterations} iterations)"

            plot_multiple_best_responses_by_time(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
        else:
            # Default behavior: create both plots
            # 1. Non-sampling vs iterations
            print(f"Searching for non-sampling algorithms for {args.game} with {args.iterations} iterations...")
            non_sampling_files = find_best_response_files(args.game, args.iterations, include_sampling=False)

            if non_sampling_files:
                print(f"Found non-sampling files ({len(non_sampling_files)}):")
                for f in non_sampling_files:
                    print(f"  - {f}")

                output_path = args.output or os.path.join(output_dir, f"{args.game}_{args.iterations}_comparison.png")
                title = args.title or f"Exploitability Comparison ({args.game}, {args.iterations} iterations)"

                plot_multiple_best_responses(
                    non_sampling_files,
                    output_path=output_path,
                    title=title,
                    log_log=not args.no_log_log
                )
            else:
                print(f"No non-sampling algorithms found for {args.game} with {args.iterations} iterations")

            # 2. All vs time
            print(f"\nSearching for all algorithms for {args.game} with {args.iterations} iterations (time plot)...")
            all_files = find_best_response_files(args.game, args.iterations, include_sampling=True)

            if all_files:
                print(f"Found files for time plot ({len(all_files)}):")
                for f in all_files:
                    print(f"  - {f}")

                output_path_time = os.path.join(output_dir, f"{args.game}_{args.iterations}_time_comparison.png")
                title_time = f"Exploitability vs Time Comparison ({args.game}, {args.iterations} iterations)"

                plot_multiple_best_responses_by_time(
                    all_files,
                    output_path=output_path_time,
                    title=title_time,
                    log_log=not args.no_log_log
                )
            else:
                print(f"No files found for time plot")

    elif args.files:
        # Legacy mode: use provided files
        valid_files = []
        for filepath in args.files:
            if os.path.exists(filepath):
                valid_files.append(filepath)
            else:
                print(f"Warning: File not found: {filepath}")

        if not valid_files:
            print("Error: No valid files found")
            return

        output_path = args.output or 'comparison.png'
        title = args.title or "Exploitability Comparison"

        if args.time:
            # Manual time comparison
            if not output_path.endswith('.png'):
                output_path = output_path.replace('.png', '_time.png')
            if output_path == 'comparison.png':
                output_path = 'comparison_time.png'
            title = args.title or "Exploitability vs Time Comparison"

            plot_manual_time_comparison(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
        else:
            # Default: plot vs iterations
            plot_multiple_best_responses(
                valid_files,
                output_path=output_path,
                title=title,
                log_log=not args.no_log_log
            )
    else:
        print("Error: Please specify either --game and --iterations or provide files as arguments")
        parser.print_help()
        return


if __name__ == "__main__":
    main()
