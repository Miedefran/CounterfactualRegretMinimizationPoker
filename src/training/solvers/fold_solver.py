import pickle as pkl
import gzip
import time
from typing import Any

from utils.data_models import KeyGenerator
from training.registry import TrainingSolver
from training.config import TrainingConfig


class AlwaysFoldSolver(TrainingSolver):

    def __init__(self, game, combination_generator):
        self.game = game
        self.combination_generator = combination_generator
        self.combinations = combination_generator.get_all_combinations()
        self.average_strategy = {}
        self.iteration_count = 1
        self.training_time = 0

        # Keep these to maintain compatibility with save/load format
        self.regret_sum = {}
        self.strategy_sum = {}

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'fold'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return AlwaysFoldSolver(game, combo_gen)

    @staticmethod
    def supports_alternating_updates() -> bool:
        return False

    @staticmethod
    def supports_partial_pruning() -> bool:
        return False

    @staticmethod
    def supports_squared_weights() -> bool:
        return False

    def train(self, iterations=1, br_tracker=None, print_interval=100, stop_exploitability_mb=None):
        """
        Traverses the entire game tree once to populate average_strategy
        with "Always Fold" (or Check if Fold is illegal) logic.
        """
        start_time = time.time()
        print("Generating Always Fold strategy...")

        # We only need one pass over all combinations to cover the whole tree
        for i, combination in enumerate(self.combinations):
            self.combination_generator.setup_game_with_combination(self.game, combination)

            # Traverse for the current state (covers both players as we walk the tree)
            self.traverse_and_populate()

            if i % 100 == 0:
                print(f"Processed {i}/{len(self.combinations)} combinations")

        self.training_time = time.time() - start_time
        print(f"Strategy generation complete in {self.training_time:.2f} seconds")

    def traverse_and_populate(self):
        if self.game.done:
            return

        current_player = self.game.current_player
        legal_actions = self.game.get_legal_actions()
        info_set_key = KeyGenerator.get_info_set_key(self.game, current_player)

        # 1. Determine the "Always Fold" move
        chosen_action = None
        if 'fold' in legal_actions:
            chosen_action = 'fold'
        elif 'check' in legal_actions:
            chosen_action = 'check'
        else:
            # Fallback for weird states (e.g. forced bet?): choose first action
            chosen_action = legal_actions[0]

        # 2. Build the strategy distribution
        strategy = {a: 0.0 for a in legal_actions}
        strategy[chosen_action] = 1.0

        # 3. Store into average_strategy
        # We perform a safe update: if we visit this infoset multiple times
        # (via different paths/permutations), the strategy remains the same.
        self.average_strategy[info_set_key] = strategy

        # 4. Recurse down ALL branches
        # Auch bei Fold: Call/Raise-Zweige traversieren
        # to populate the strategy for deeper nodes. This ensures the
        # Best Response calculator doesn't fall back to Uniform Random
        # if it queries a "what-if" scenario.
        for action in legal_actions:
            self.game.step(action)
            self.traverse_and_populate()
            self.game.step_back()

    def get_average_strategy(self):
        return self.average_strategy

    """Storage Methods (Copied for compatibility)"""

    def save_gzip(self, filepath):
        data = {
            'regret_sum': self.regret_sum,
            'strategy_sum': self.strategy_sum,
            'average_strategy': self.average_strategy,
            'iteration_count': self.iteration_count,
            'training_time': self.training_time
        }

        with gzip.open(filepath, 'wb') as f:
            pkl.dump(data, f)

        print(f"Saved to {filepath}")
