"""CFR+ with pre-built game tree. Regret Matching+ and linear strategy averaging."""

import gzip
import pickle as pkl
from typing import Any

from training.solvers.cfr_solver_with_tree import CFRSolverWithTree
from training.registry import TrainingSolver
from training.config import TrainingConfig


class CFRPlusWithTree(CFRSolverWithTree):
    """
    CFR+ with pre-built game tree.
    - Clamp negative regrets to zero after each player pass.
    - Linear strategy averaging (t or t²).
    """

    def __init__(
            self,
            game,
            combination_generator,
            game_name=None,
            load_tree=True,
            alternating_updates=True,
            partial_pruning=False,
            squared_weight=False,
    ):
        super().__init__(
            game,
            combination_generator,
            game_name=game_name,
            load_tree=load_tree,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self.squared_weight = squared_weight

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'cfr_plus_with_tree'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return CFRPlusWithTree(
            game,
            combo_gen,
            game_name=config.game,
            alternating_updates=config.alternating_updates,
            partial_pruning=config.partial_pruning,
            squared_weight=config.squared_weight,
        )

    @staticmethod
    def supports_squared_weights() -> bool:
        return True

    def _get_strategy_sum_weight(self):
        t = float(self._current_iteration)
        return t ** 2 if self.squared_weight else t

    def after_player_traversal(self, player: int):
        self._clamp_regret_sum_non_negative()

    def after_simultaneous_traversal(self):
        self._clamp_regret_sum_non_negative()

    def _clamp_regret_sum_non_negative(self):
        """Set all negative cumulative regrets to zero."""
        for info_state in self.regret_sum:
            for action in self.regret_sum[info_state]:
                if self.regret_sum[info_state][action] < 0:
                    self.regret_sum[info_state][action] = 0.0

    def save_gzip(self, filepath):
        """Save solver state to gzip file."""
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

    def load_gzip(self, filepath):
        """Load solver state from gzip file."""
        with gzip.open(filepath, 'rb') as f:
            data = pkl.load(f)

        self.regret_sum = data['regret_sum']
        self.strategy_sum = data['strategy_sum']
        self.average_strategy = data.get('average_strategy', {})
        self.iteration_count = data.get('iteration_count', 0)
        self.training_time = data.get('training_time', 0)
        self._current_iteration = max(1, self.iteration_count + 1)

        # Rebuild policy cache
        self._policy_cache = {}
        for info_state in self.regret_sum.keys():
            self._get_policy(info_state)

        print(f"Loaded from {filepath}")
