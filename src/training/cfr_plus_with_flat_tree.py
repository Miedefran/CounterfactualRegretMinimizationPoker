"""
CFR+ auf Flat-Tree Basis.

Diese Datei enthält bewusst nur die CFR+-spezifischen Overwrites, damit der Kern
(`CFRSolverWithFlatTree`) lesbar bleibt.
"""

import numpy as np

from training.cfr_solver_with_flat_tree import CFRSolverWithFlatTree


class CFRPlusWithFlatTree(CFRSolverWithFlatTree):
    """
    CFR+ variant:
    - clamp negative regrets after each traversal pass
    - linear averaging: strategy_sum adds t * reach * sigma
    """

    def _get_strategy_sum_weight(self) -> float:
        # Linear averaging weight (1-indexed)
        return float(self._current_iteration)

    def after_player_traversal(self, player: int):
        self.regret_sum = np.maximum(self.regret_sum, 0.0)

    def after_simultaneous_traversal(self):
        self.regret_sum = np.maximum(self.regret_sum, 0.0)

