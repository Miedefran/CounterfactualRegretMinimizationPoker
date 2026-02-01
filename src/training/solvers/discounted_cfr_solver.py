import pickle as pkl
import gzip
import numpy as np
from typing import Any

from training.solvers.cfr_solver import CFRSolver
from training.registry import TrainingSolver
from training.config import TrainingConfig


class DiscountedCFRSolver(CFRSolver):
    """
    DCFR variant (dynamic traversal).
    - Discount positive regrets by t^alpha/(t^alpha+1), negative by t^beta/(t^beta+1) after each pass.
    - Strategy sum weight t^gamma.
    """

    def __init__(self, game, combination_generator, alternating_updates=True, partial_pruning=False,
                 alpha=1.5, beta=0.0, gamma=2.0):
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        print(f"Discounted CFR initialized with α={alpha}, β={beta}, γ={gamma}")

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'discounted_cfr'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return DiscountedCFRSolver(game, combo_gen, alternating_updates=config.alternating_updates,
                                   partial_pruning=config.partial_pruning, alpha=config.dcfr_alpha,
                                   beta=config.dcfr_beta, gamma=config.dcfr_gamma)

    def _get_strategy_sum_weight(self):
        return float(self._current_iteration) ** float(self.gamma)

    def after_player_traversal(self, player: int):
        self._apply_regret_discounting(player=player)

    def after_simultaneous_traversal(self):
        self._apply_regret_discounting(player=None)

    def _apply_regret_discounting(self, player):
        """
        Apply alpha/beta discounting to cumulative regrets (after traversal).
        
        Args:
            player: Player to apply discounting for (0, 1, or None for both)
        """
        t = self._current_iteration

        if self.alpha > 0:
            t_alpha = t ** self.alpha
            positive_discount = t_alpha / (t_alpha + 1)
        else:
            positive_discount = 1.0

        if self.beta == 0:
            negative_discount = 0.5
        elif self.beta > 0:
            t_beta = t ** self.beta
            negative_discount = t_beta / (t_beta + 1)
        else:
            negative_discount = 1.0

        for info_state, regret_dict in self.regret_sum.items():
            if len(info_state) >= 4:
                info_set_player = info_state[3]
            else:
                continue

            if player is not None and info_set_player != player:
                continue

            for action in regret_dict.keys():
                current_regret = self.regret_sum[info_state][action]
                if current_regret >= 0:
                    self.regret_sum[info_state][action] *= positive_discount
                else:
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

        self._current_iteration = max(1, self.iteration_count + 1)

        self._policy_cache = {}
        for info_set_key in self.regret_sum.keys():
            if info_set_key in self.strategy_sum:
                legal_actions = list(self.strategy_sum[info_set_key].keys())
                if legal_actions:
                    self._get_policy(info_set_key, legal_actions)

        print(f"Loaded from {filepath}")
