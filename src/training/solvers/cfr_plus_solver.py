import numpy as np
from typing import Any
from training.solvers.cfr_solver import CFRSolver
from training.registry import TrainingSolver
from training.config import TrainingConfig


class CFRPlusSolver(CFRSolver):
    """
    CFR+ variant (dynamic traversal).
    - Clamp negative regrets to zero after each player pass.
    - Linear strategy averaging: strategy_sum adds t * reach * sigma (or t² if squared_weight).
    """

    def __init__(self, game, combination_generator, alternating_updates=True, partial_pruning=False,
                 squared_weight=False):
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self.squared_weight = squared_weight

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'cfr_plus'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return CFRPlusSolver(game, combo_gen, alternating_updates=config.alternating_updates,
                             partial_pruning=config.partial_pruning, squared_weight=config.squared_weight)

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
