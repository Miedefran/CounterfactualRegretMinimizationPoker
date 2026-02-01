"""
Discounted CFR (DCFR) on flat tree basis.

Based on `training/discounted_cfr_solver_with_tree.py`:
- Regrets are discounted after each pass (positive with alpha, negative with beta).
- StrategySum is weighted with t^gamma (t^γ weighting of the new contribution).
"""

from __future__ import annotations

import gzip
import pickle as pkl
from typing import Optional, Any

from training.solvers.cfr_solver_with_flat_tree import CFRSolverWithFlatTree
from training.registry import TrainingSolver
from training.config import TrainingConfig


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

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'discounted_cfr_with_flat_tree'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return DiscountedCFRWithFlatTree(
            game,
            combo_gen,
            game_name=config.game,
            alpha=config.dcfr_alpha,
            beta=config.dcfr_beta,
            gamma=config.dcfr_gamma,
            alternating_updates=config.alternating_updates,
            partial_pruning=config.partial_pruning,
        )

    def _get_strategy_sum_weight(self) -> float:
        # Weight new contribution with t^gamma
        t = float(self._current_iteration)
        return t ** float(self.gamma)

    def after_player_traversal(self, player: int):
        self._apply_regret_discounting(player=player)

    def after_simultaneous_traversal(self):
        self._apply_regret_discounting(player=None)

    def _apply_regret_discounting(self, player: Optional[int]):
        """
        Apply discounting to cumulative regrets after update.
        For alternating: only infosets of player.
        
        Discounting formulas:
        - Positive regrets: t^alpha / (t^alpha + 1)
        - Negative regrets: t^beta / (t^beta + 1) if beta > 0, else 0.5 if beta == 0
        """
        t = float(self._current_iteration)

        # Positive regret discounting: t^alpha / (t^alpha + 1)
        if self.alpha > 0:
            t_alpha = t ** float(self.alpha)
            positive_discount = t_alpha / (t_alpha + 1.0)
        else:
            positive_discount = 1.0

        # Negative regret discounting: t^beta / (t^beta + 1) or 0.5 if beta == 0
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
