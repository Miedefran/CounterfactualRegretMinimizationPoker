"""
CFR+ on flat tree basis.

This file intentionally contains only CFR+-specific overrides, keeping the core
(`CFRSolverWithFlatTree`) readable.
"""

import numpy as np
from typing import Any

from training.solvers.cfr_solver_with_flat_tree import CFRSolverWithFlatTree
from training.registry import TrainingSolver
from training.config import TrainingConfig


class CFRPlusWithFlatTree(CFRSolverWithFlatTree):
    """
    CFR+ variant:
    - clamp negative regrets after each traversal pass
    - linear averaging: strategy_sum adds t * reach * sigma (optional squared weight: t² instead of t)
    """

    def __init__(self, *args, squared_weight=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.squared_weight = squared_weight

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'cfr_plus_with_flat_tree'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return CFRPlusWithFlatTree(
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

    def _get_strategy_sum_weight(self) -> float:
        # Linear averaging weight (1-indexed). With --squared-weight: t² instead of t
        t = float(self._current_iteration)
        return t ** 2 if self.squared_weight else t

    def after_player_traversal(self, player: int):
        self.regret_sum = np.maximum(self.regret_sum, 0.0)

    def after_simultaneous_traversal(self):
        self.regret_sum = np.maximum(self.regret_sum, 0.0)
