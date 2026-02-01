"""Discounted CFR with pre-built game tree. Same discounting as DCFR (alpha, beta, gamma)."""

import gzip
import pickle as pkl
from typing import Any

from training.solvers.cfr_solver_with_tree import CFRSolverWithTree
from training.registry import TrainingSolver
from training.config import TrainingConfig


class DiscountedCFRWithTreeSolver(CFRSolverWithTree):
    """
    DCFR with pre-built game tree.
    - Discount positive regrets by t^alpha/(t^alpha+1), negative by t^beta/(t^beta+1) after each pass.
    - Strategy sum weight t^gamma.
    """

    def __init__(
            self,
            game,
            combination_generator,
            game_name=None,
            load_tree=True,
            alpha=1.5,
            beta=0.0,
            gamma=2.0,
            alternating_updates=True,
            partial_pruning=False,
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

        print(f"Discounted CFR with Tree initialized with α={alpha}, β={beta}, γ={gamma}")

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'discounted_cfr_with_tree'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return DiscountedCFRWithTreeSolver(
            game, combo_gen,
            game_name=config.game,
            alpha=config.dcfr_alpha,
            beta=config.dcfr_beta,
            gamma=config.dcfr_gamma,
            alternating_updates=config.alternating_updates,
            partial_pruning=config.partial_pruning,
        )

    def after_player_traversal(self, player: int):
        # DCFR: apply discounting after each player update (before policy update)
        self._apply_regret_discounting(player=player)

    def after_simultaneous_traversal(self):
        self._apply_regret_discounting(player=None)

    def _get_strategy_sum_weight(self):
        return float(self._current_iteration) ** float(self.gamma)

    def _apply_regret_discounting(self, player):
        """
        Apply alpha/beta discounting to cumulative regrets (after traversal).
        
        Args:
            player: Player to apply discounting for (0, 1, or None for both)
        """
        t = self._current_iteration

        # Discount factors for alpha/beta
        if self.alpha > 0:
            t_alpha = t ** self.alpha
            positive_discount = t_alpha / (t_alpha + 1)
        else:
            positive_discount = 1.0

        # For beta=0 use 0.5; for beta>0 use t^beta/(t^beta+1)
        if self.beta == 0:
            negative_discount = 0.5
        elif self.beta > 0:
            t_beta = t ** self.beta
            negative_discount = t_beta / (t_beta + 1)
        else:
            # For beta < 0: no discounting
            negative_discount = 1.0

        # Apply discounting to all infosets
        for info_state, regret_dict in self.regret_sum.items():
            # Skip if not for this player
            node_ids = self.infoset_to_nodes.get(info_state, [])
            if not node_ids:
                continue

            node = self.nodes[node_ids[0]]
            if player is not None and node.player != player:
                continue

            # Apply discounting based on sign
            for action in regret_dict.keys():
                current_regret = self.regret_sum[info_state][action]
                if current_regret >= 0:
                    # Positive regret: apply alpha discount
                    self.regret_sum[info_state][action] *= positive_discount
                else:
                    # Negative regret: apply beta discount
                    self.regret_sum[info_state][action] *= negative_discount

    def save_gzip(self, filepath):
        """Save solver state to gzip file."""
        data = {
            'regret_sum': self.regret_sum,
            'strategy_sum': self.strategy_sum,
            'average_strategy': self.average_strategy,
            'iteration_count': self.iteration_count,
            'training_time': self.training_time,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
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
        self.alpha = data.get('alpha', 1.5)
        self.beta = data.get('beta', 0.0)
        self.gamma = data.get('gamma', 2.0)

        # _current_iteration is 1-indexed; iteration_count is 0-indexed
        self._current_iteration = max(1, self.iteration_count + 1)

        # Rebuild policy cache
        self._policy_cache = {}
        for info_state in self.regret_sum.keys():
            self._get_policy(info_state)

        print(f"Loaded from {filepath}")
