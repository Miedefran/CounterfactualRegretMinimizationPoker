"""
Discounted CFR (DCFR) auf Flat-Tree Basis.

Orientiert sich an `training/discounted_cfr_solver_with_tree.py`:
- Regrets werden nach jedem Pass discountet (positive mit alpha, negative mit beta).
- StrategySum wird mit t^gamma gewichtet (OpenSpiel-style).
"""

from __future__ import annotations

import gzip
import pickle as pkl
from typing import Optional

import numpy as np

from training.cfr_solver_with_flat_tree import CFRSolverWithFlatTree


class DiscountedCFRWithFlatTree(CFRSolverWithFlatTree):
    """
    Discounted CFR (DCFR) on a flat tree.

    Matching `DiscountedCFRWithTreeSolver`:
    - Regrets are discounted after each traversal pass:
      positive regrets  *= t^alpha / (t^alpha + 1)
      negative regrets  *= t^beta  / (t^beta  + 1)  (beta=0 => 1/2)
    - Strategy sum uses t^gamma weighting for the newly added contribution.
    """

    def __init__(
        self,
        game,
        combination_generator,
        game_name: Optional[str] = None,
        load_tree: bool = True,
        alpha: float = 1.5,
        beta: float = 0.0,
        gamma: float = 2.0,
        alternating_updates: bool = True,
        partial_pruning: bool = False,
    ):
        super().__init__(
            game,
            combination_generator,
            game_name=game_name,
            load_tree=load_tree,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        print(f"Discounted CFR with Flat Tree initialized with α={alpha}, β={beta}, γ={gamma}")

    def _get_strategy_sum_weight(self) -> float:
        # OpenSpiel-style: multiply the new contribution with t^gamma
        t = float(self._current_iteration)
        return t ** float(self.gamma)

    def after_player_traversal(self, player: int):
        self._apply_regret_discounting(player=player)

    def after_simultaneous_traversal(self):
        self._apply_regret_discounting(player=None)

    def _apply_regret_discounting(self, player: Optional[int]):
        """
        Apply discounting to the cumulative regrets AFTER adding the instantaneous regrets.

        Implementation note:
        - We only discount infosets of `player` in alternating mode (like the tree solver).
        - We use `flat.infoset_player[iid]` to select infosets quickly.
        """
        t = float(self._current_iteration)

        if self.alpha > 0:
            t_alpha = t ** float(self.alpha)
            positive_discount = t_alpha / (t_alpha + 1.0)
        else:
            positive_discount = 1.0

        # Beta handling (as in your existing DCFR solvers)
        if self.beta == 0:
            negative_discount = 0.5
        elif self.beta > 0:
            t_beta = t ** float(self.beta)
            negative_discount = t_beta / (t_beta + 1.0)
        else:
            negative_discount = 1.0

        if player is None:
            mask = slice(None)
        else:
            mask = (self.flat.infoset_player == int(player))

        r = self.regret_sum[mask]
        pos = (r >= 0.0)
        r[pos] *= positive_discount
        r[~pos] *= negative_discount
        self.regret_sum[mask] = r

    def save_gzip(self, filepath):
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "average_strategy": self.average_strategy,
            "iteration_count": self.iteration_count,
            "training_time": self.training_time,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }
        with gzip.open(filepath, "wb") as f:
            pkl.dump(data, f)
        print(f"Saved to {filepath}")

    def load_gzip(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = pkl.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.average_strategy = data.get("average_strategy", {})
        self.iteration_count = data.get("iteration_count", 0)
        self.training_time = data.get("training_time", 0.0)
        self.alpha = data.get("alpha", 1.5)
        self.beta = data.get("beta", 0.0)
        self.gamma = data.get("gamma", 2.0)
        print(f"Loaded from {filepath}")

