"""
Best Response Evaluator for CFR Training.

This file provides functions to compute and store best response values during training.
"""

import os
import gzip
import pickle
import time
import json
import math
from typing import Type

from training.registry import TrainingGame
from utils.poker_utils import find_game_class_for_abstraction
from training.progress_reporter import ProgressReporter


def get_public_state_tree_path(game_name, abstract_suits=None):
    """
    Get path to Public State Tree for a given game.

    Args:
        game_name: Name of the game (e.g. 'leduc', 'kuhn_case1', etc.)
        abstract_suits: If None, automatically determined (default: True for leduc/twelve_card_poker)
                        If True, use standard name
                        If False, use name with "_NOT_abstracted"

    Returns:
        Path to Public State Tree file
    """
    # Normalize game_name for filename
    # For kuhn_caseX we use 'kuhn'
    if game_name.startswith('kuhn'):
        save_name = 'kuhn'
    else:
        save_name = game_name

    # Determine if suit abstraction should be used
    # Default for leduc and twelve_card_poker
    if abstract_suits is None:
        abstract_suits = (game_name in ['leduc', 'twelve_card_poker'])

    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(script_dir, 'data', 'trees', 'public_state_trees')

    # If abstract_suits=False, add "_NOT_abstracted" to the name
    if abstract_suits:
        filename = f"{save_name}_public_tree_v2.pkl.gz"
    else:
        filename = f"{save_name}_public_tree_v2_NOT_abstracted.pkl.gz"

    path = os.path.join(base_dir, filename)

    return path


def load_schedule_config(config_path=None):
    """
    Load schedule configuration from a JSON file.

    Args:
        config_path: Path to JSON file or name of a predefined schedule

    Returns:
        Dictionary with schedule configuration or None
    """
    if config_path is None:
        return None

    if config_path.endswith('.json'):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Schedule config not found: {config_path}")
    else:
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config',
            'br_eval_schedules.json'
        )
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                all_schedules = json.load(f)
                if config_path in all_schedules:
                    return all_schedules[config_path]
                else:
                    raise ValueError(f"Schedule '{config_path}' not found in {default_config_path}")
        else:
            raise FileNotFoundError(f"Default schedule config not found: {default_config_path}")


class BestResponseTracker:
    """
    Manage best response values during training.
    The Public State Tree is loaded once during initialization and then reused.
    """

    def __init__(self, game_name, game_class: Type[TrainingGame], schedule_config=None,
                 progress_reporter: ProgressReporter =None):
        """
        Args:
            game_name: Name of the game
            game_class: Class of the game
            schedule_config: Schedule configuration (Dict, JSON path, or schedule name)
                           If None or Integer, fixed interval is used (backward compatibility)
            progress_reporter: Optional ProgressReporter for TUI integration
        """
        self.game_name = game_name
        self.game_class: Type[TrainingGame] = game_class
        # Store use_suit_abstraction for later
        self.use_suit_abstraction = game_class.suit_abstraction
        self.values = []  # List of (iteration, exploitability_mb, br_value_p0, br_value_p1, cumulative_training_time)
        self.total_br_time = 0.0  # Cumulative time for best response evaluations
        self.last_eval_iteration = 0
        self.training_start_time = None  # Training start time
        self.progress_reporter: ProgressReporter = progress_reporter  # Optional TUI progress reporter

        if schedule_config is None:
            schedule_config = {"type": "fixed", "interval": 100}
        elif isinstance(schedule_config, int):
            schedule_config = {"type": "fixed", "interval": schedule_config}
        elif isinstance(schedule_config, str):
            schedule_config = load_schedule_config(schedule_config)

        self.schedule_config = schedule_config
        self.schedule_type = schedule_config.get("type", "fixed")

        if self.schedule_type == "fixed":
            self.interval = schedule_config.get("interval", 100)
        elif self.schedule_type == "logarithmic":
            self.base_interval = schedule_config.get("base_interval", 10)
            self.target_iteration = schedule_config.get("target_iteration", 100)
            self.target_interval = schedule_config.get("target_interval", 100)
            # unbounded: If True, interval continues to grow after target_iteration
            # target_iteration/target_interval define growth curve (how fast it grows)
            self.unbounded = schedule_config.get("unbounded", False)
        elif self.schedule_type == "custom":
            self.custom_schedule = schedule_config.get("schedule", [])
            if not self.custom_schedule:
                raise ValueError("Custom schedule requires 'schedule' list")
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Lade Public State Tree einmal beim Initialisieren
        self.tree = None
        self._load_tree()

    def to_milliblinds_per_game(self, value):
        game = self.game_class()
        big_blind_equiv = game.get_big_blind_equivalent()
        return (value / big_blind_equiv) * 1000.0


    def evaluate_best_response(self, game_name, tree, average_strategy, iteration):
        """
        Compute best response value for both players.

        Args:
            game_name: Name of the game
            tree: Public State Tree (already loaded)
            average_strategy: Dictionary with current average strategy
            iteration: Current iteration number

        Returns:
            Tuple (br_value_p0, br_value_p1, elapsed_time) or None on error
        """
        try:
            start_time = time.time()

            # Import best response functions
            from evaluation.best_response_agent_v2 import compute_best_response_value

            # Compute best response for both players
            br_value_p0 = compute_best_response_value(game_name, 0, tree, average_strategy)
            br_value_p1 = compute_best_response_value(game_name, 1, tree, average_strategy)

            elapsed_time = time.time() - start_time

            # Exploitability = (BR value P0 + BR value P1) / 2
            exploitability = (br_value_p0 + br_value_p1) / 2.0

            # Convert to milliblinds per game
            br_value_p0_mb = self.to_milliblinds_per_game(br_value_p0)
            br_value_p1_mb = self.to_milliblinds_per_game(br_value_p1)
            exploitability_mb = self.to_milliblinds_per_game(exploitability)

            print(
                f"  Best Response @ Iteration {iteration}: P0={br_value_p0_mb:.6f} mb/g, P1={br_value_p1_mb:.6f} mb/g, Exploitability={exploitability_mb:.6f} mb/g (took {elapsed_time:.2f}s)")

            return (br_value_p0, br_value_p1, elapsed_time)

        except Exception as e:
            print(f"WARNING: Fehler bei Best Response Evaluation: {e}")
            return None

    def _load_tree(self):
        """Load Public State Tree once during initialization."""
        try:
            from evaluation.best_response_agent_v2 import load_public_tree
            from evaluation.build_public_state_tree_v2 import build_public_state_tree, save_public_state_tree
            from utils.poker_utils import find_game_class_for_abstraction

            # Use stored use_suit_abstraction setting
            use_suit_abstraction = self.use_suit_abstraction

            # Try to load standard path (with abstract_suits parameter)
            tree_path = get_public_state_tree_path(self.game_name, abstract_suits=use_suit_abstraction)
            if not os.path.exists(tree_path):
                # Tree does not exist - build it automatically
                print(f"Public State Tree not found. Building it automatically...")

                # Determine game class
                game_class = find_game_class_for_abstraction(self.game_name, use_suit_abstraction)
                if game_class is None:
                    print(f"WARNING: Unknown game: {self.game_name}")
                    print("Best Response Evaluation will be disabled")
                    return

                # Determine save_name
                if self.game_name.startswith('kuhn'):
                    save_name = 'kuhn'
                else:
                    save_name = self.game_name

                # Build and save tree
                tree = build_public_state_tree(game_class, use_cache=True, abstract_suits=use_suit_abstraction)
                tree_path = save_public_state_tree(save_name, tree, abstract_suits=use_suit_abstraction)
                print(f"Public State Tree successfully built and saved.")

            self.tree = load_public_tree(tree_path)
            abstraction_str = " (suit abstracted)" if use_suit_abstraction else ""
            print(f"Public State Tree loaded{abstraction_str}: {tree_path}")

        except Exception as e:
            print(f"WARNING: Error loading Public State Tree: {e}")
            import traceback
            traceback.print_exc()
            print("Best Response Evaluation will be disabled")

    def get_current_interval(self, iteration):
        """
        Compute current evaluation interval based on iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Interval for this iteration
        """
        if self.schedule_type == "fixed":
            return self.interval

        elif self.schedule_type == "logarithmic":
            if iteration <= self.base_interval:
                return self.base_interval

            log_factor = math.log(iteration / self.base_interval + 1)
            target_log = math.log(self.target_iteration / self.base_interval + 1)

            if target_log == 0:
                return self.base_interval

            interval = self.base_interval + (self.target_interval - self.base_interval) * (log_factor / target_log)

            # If unbounded: No upper limit, interval continues to grow
            if self.unbounded:
                return max(self.base_interval, int(interval))
            else:
                return max(self.base_interval, min(int(interval), self.target_interval))

        elif self.schedule_type == "custom":
            for i, (threshold, interval) in enumerate(self.custom_schedule):
                if i == len(self.custom_schedule) - 1 or iteration < self.custom_schedule[i + 1][0]:
                    return interval
            return self.custom_schedule[-1][1]

        return 100

    def should_evaluate(self, iteration):
        """
        Check if evaluation should occur at this iteration.

        Args:
            iteration: Current iteration number

        Returns:
            True if evaluation should occur, else False
        """
        # Design decision: We always evaluate at iteration 1, regardless of schedule type.
        # This makes runs more comparable (always a "baseline" measurement point) and prevents
        # a schedule with large interval from setting the first BR point very late.
        if iteration == 1:
            return True

        if self.schedule_type == "fixed":
            return iteration % self.interval == 0

        if self.schedule_type == "custom":
            if iteration < self.custom_schedule[0][0]:
                return False

        current_interval = self.get_current_interval(iteration)
        return (iteration - self.last_eval_iteration) >= current_interval

    def add_value(self, iteration, exploitability_mb, br_value_p0, br_value_p1, cumulative_training_time=None,
                  eval_time=0.0):
        """
        Add an exploitability value and BR values.

        Args:
            iteration: Iteration number
            exploitability_mb: Exploitability in milliblinds per game
            br_value_p0: Best response value for player 0 (raw)
            br_value_p1: Best response value for player 1 (raw)
            cumulative_training_time: Cumulative training time up to this iteration (without BR time)
            eval_time: Time taken for this BR evaluation (seconds)
        """
        if cumulative_training_time is None:
            cumulative_training_time = 0.0
        self.values.append((iteration, exploitability_mb, br_value_p0, br_value_p1, cumulative_training_time))

        # Report to progress reporter if available
        if self.progress_reporter is not None:
            br_value_p0_mb = self.to_milliblinds_per_game(br_value_p0)
            br_value_p1_mb = self.to_milliblinds_per_game(br_value_p1)
            self.progress_reporter.add_br_value(
                iteration=iteration,
                br_value_p0_mb=br_value_p0_mb,
                br_value_p1_mb=br_value_p1_mb,
                exploitability_mb=exploitability_mb,
                eval_time=eval_time,
                cumulative_training_time=cumulative_training_time,
            )
            # Also update iteration count
            self.progress_reporter.update_iteration(iteration)

    def evaluate_and_add(self, average_strategy, iteration, cumulative_training_time=None, start_time=None):
        """
        Compute best response value and add it.

        Args:
            average_strategy: Current average strategy
            iteration: Current iteration number
            cumulative_training_time: Cumulative training time up to this iteration (without BR time).
                                     If None and start_time given, automatically computed.
            start_time: Training start time (optional, for automatic time calculation)

        Returns:
            Time consumed for evaluation (in seconds)
        """
        if self.tree is None:
            # Tree could not be loaded, skip evaluation
            return 0.0

        # Compute time BEFORE evaluation (without current BR evaluation)
        # total_br_time contains all previous BR evaluations
        if cumulative_training_time is None and start_time is not None:
            import time
            # Time WITHOUT all previous BR evaluations
            cumulative_training_time = time.time() - start_time - self.total_br_time

        # Perform best response evaluation
        result = self.evaluate_best_response(self.game_name, self.tree, average_strategy, iteration)
        if result is not None:
            br_value_p0, br_value_p1, elapsed_time = result
            # Update total_br_time AFTER time calculation
            self.total_br_time += elapsed_time

            # Exploitability = (BR value P0 + BR value P1) / 2
            exploitability = (br_value_p0 + br_value_p1) / 2.0
            exploitability_mb = self.to_milliblinds_per_game(exploitability)
            # Store time WITHOUT BR evaluations
            self.add_value(iteration, exploitability_mb, br_value_p0, br_value_p1, cumulative_training_time,
                           eval_time=elapsed_time)

            return elapsed_time

        return 0.0

    def get_total_br_time(self):
        """
        Return cumulative time used for best response evaluations.

        Returns:
            Total time in seconds
        """
        return self.total_br_time

    def plot(self, output_path=None, log_scale=True, log_log=True):
        """
        Plot exploitability over iterations.

        Args:
            output_path: Optional path to save plot
            log_scale: If True, X-axis is logarithmically scaled (default: True)
            log_log: If True, BOTH axes are logarithmically scaled (log-log plot, default: True)
        """
        if not self.values:
            print("No exploitability values available for plotting")
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            iterations = [x[0] for x in self.values]
            exploitability = [x[1] for x in self.values]  # Exploitability

            plt.figure(figsize=(12, 8))
            plt.plot(iterations, exploitability, label='Exploitability', marker='o', linewidth=2, markersize=6)

            # X-Achse
            if log_scale or log_log:
                plt.xscale('log')
                plt.xlabel('Iterations (log scale)')
            else:
                plt.xlabel('Iterations')

            # Y-Achse
            if log_log:
                plt.yscale('log')
                plt.ylabel('Exploitability (mb/g, log scale)')
            else:
                plt.ylabel('Exploitability (mb/g)')

            plt.title(f'Exploitability During Training ({self.game_name})')
            plt.legend()
            plt.grid(True, alpha=0.3, which='both')  # 'both' for log and linear grid

            # Set meaningful X-axis ticks for log scale
            if (log_scale or log_log) and iterations:
                max_iter = max(iterations)
                # Create logarithmic ticks
                if max_iter <= 100:
                    ticks = [1, 2, 5, 10, 20, 50, 100]
                elif max_iter <= 1000:
                    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
                elif max_iter <= 10000:
                    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
                else:
                    ticks = [1, 10, 100, 1000, 10000, 100000]

                # Filter ticks larger than max_iter
                ticks = [t for t in ticks if t <= max_iter]
                plt.xticks(ticks, [str(t) for t in ticks])

            # Set meaningful Y-axis ticks for log scale
            if log_log and exploitability:
                # Filter negative or zero values
                positive_values = [v for v in exploitability if v > 0]
                if positive_values:
                    min_exp = min(positive_values)
                    max_exp = max(positive_values)

                    # Create logarithmic ticks based on value range
                    if max_exp <= 1:
                        y_ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
                    elif max_exp <= 10:
                        y_ticks = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
                    elif max_exp <= 100:
                        y_ticks = [1, 2, 5, 10, 20, 50, 100]
                    elif max_exp <= 1000:
                        y_ticks = [10, 20, 50, 100, 200, 500, 1000]
                    else:
                        y_ticks = [1, 10, 100, 1000, 10000]

                    # Filter ticks outside value range
                    y_ticks = [t for t in y_ticks if min_exp <= t <= max_exp * 1.1]
                    if y_ticks:
                        plt.yticks(y_ticks,
                                   [f'{t:.2f}' if t < 1 else f'{int(t) if t == int(t) else t}' for t in y_ticks])

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {output_path}")
            else:
                plt.show()

        except ImportError:
            print("WARNING: matplotlib not available, plot will be skipped")
        except Exception as e:
            print(f"WARNING: Error plotting: {e}")

    def save(self, filepath):
        """
        Save best response values to a file.

        Args:
            filepath: Path to file
        """
        data = {
            'game_name': self.game_name,
            'schedule_config': self.schedule_config,
            'values': self.values
        }

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Best response values saved: {filepath}")

    def load(self, filepath):
        """
        Load best response values from a file.

        Args:
            filepath: Path to file
        """
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.game_name = data['game_name']
        loaded_values = data['values']

        if 'schedule_config' in data:
            self.schedule_config = data['schedule_config']
            self.schedule_type = self.schedule_config.get("type", "fixed")
            if self.schedule_type == "fixed":
                self.interval = self.schedule_config.get("interval", 100)
            elif self.schedule_type == "logarithmic":
                self.base_interval = self.schedule_config.get("base_interval", 10)
                self.target_iteration = self.schedule_config.get("target_iteration", 100)
                self.target_interval = self.schedule_config.get("target_interval", 100)
            elif self.schedule_type == "custom":
                self.custom_schedule = self.schedule_config.get("schedule", [])
        else:
            eval_interval = data.get('eval_interval', 100)
            self.schedule_config = {"type": "fixed", "interval": eval_interval}
            self.schedule_type = "fixed"
            self.interval = eval_interval

        # Backward compatibility: Old files have different formats
        # Format 1: (iteration, exploitability) - very old
        # Format 2: (iteration, exploitability_mb, br_value_p0, br_value_p1) - old
        # Format 3: (iteration, exploitability_mb, br_value_p0, br_value_p1, cumulative_training_time) - new
        if loaded_values:
            if len(loaded_values[0]) == 2:
                # Old format: Convert to new format (BR values and time missing)
                self.values = [(it, exp, None, None, 0.0) for it, exp in loaded_values]
                print(f"WARNING: Old file format detected (2 values), BR values and time missing")
            elif len(loaded_values[0]) == 4:
                # Old format: Convert to new format (time missing)
                self.values = [(it, exp, br0, br1, 0.0) for it, exp, br0, br1 in loaded_values]
                print(f"WARNING: Old file format detected (4 values), time missing - set to 0")
            else:
                # New format with time
                self.values = loaded_values
        else:
            self.values = []

        print(f"Best response values loaded: {filepath}")
