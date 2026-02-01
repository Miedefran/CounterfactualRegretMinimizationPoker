"""External Sampling CFR (dynamic traversal). Sample opponent actions and chance; own actions expanded full-width."""

import random
import numpy as np
from typing import Any

from utils.data_models import KeyGenerator
from training.solvers.cfr_solver import CFRSolver
from training.registry import TrainingSolver
from training.config import TrainingConfig


class ExternalSamplingCFRSolver(CFRSolver):
    """
    External Sampling CFR (dynamic traversal).
    - Opponent nodes: sample one action (perfect recall: same action per infoset).
    - Own nodes: expand all actions.
    - Chance: sample.
    """

    def __init__(self, game, combination_generator, alternating_updates=False, partial_pruning=False):
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'external_sampling'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return ExternalSamplingCFRSolver(game, combo_gen)

    @staticmethod
    def supports_alternating_updates() -> bool:
        return False

    @staticmethod
    def supports_partial_pruning() -> bool:
        return False

    @staticmethod
    def supports_squared_weights() -> bool:
        return False

    def cfr_iteration(self):
        """One external sampling CFR iteration: one path per player (opponent/chance sampled)."""
        self._current_iteration = self.iteration_count + 1
        self._policy_cache_this_iteration = {}
        for player in range(2):
            self._sampled_actions = {}
            self._policy_updated_infosets = set()
            self.game.reset(0)
            self._update_regrets_dynamic(player, opponent_reach=1.0)

    def _update_regrets_dynamic(self, player, opponent_reach=1.0):
        """Recursive traversal with game API: sample chance/opponent, expand own nodes full-width."""
        # Terminal node: return payoff
        if self.game.done:
            return self.game.get_payoff(player)

        # Chance node: sample outcome
        if hasattr(self.game, 'is_chance_node') and self.game.is_chance_node():
            outcomes_with_probs = self.game.get_chance_outcomes_with_probs()
            if not outcomes_with_probs:
                return 0.0
            # Sample chance outcome according to probabilities
            outcomes = list(outcomes_with_probs.keys())
            weights = np.array([outcomes_with_probs.get(o, 0.0) for o in outcomes], dtype=np.float64)
            s = weights.sum()
            if s <= 0:
                outcome = random.choice(outcomes)
            else:
                weights = weights / s
                outcome = outcomes[int(np.random.choice(len(outcomes), p=weights))]
            self.game.step(outcome)
            # Recurse with sampled outcome
            value = self._update_regrets_dynamic(player, opponent_reach=opponent_reach)
            self.game.step_back()
            return value

        # Decision node: get current policy
        current_player = self.game.current_player
        legal_actions = self.game.get_legal_actions()
        info_set_key = KeyGenerator.get_info_set_key(self.game, current_player)
        self.ensure_init(info_set_key, legal_actions)
        current_policy = self.get_current_strategy(info_set_key, legal_actions)

        value = 0.0
        child_values = {}

        # Opponent's node: sample one action (perfect recall: same action per infoset)
        if current_player != player:
            if info_set_key not in self._sampled_actions:
                sampled_action = self._sample_action(current_policy, legal_actions)
                self._sampled_actions[info_set_key] = sampled_action
            else:
                sampled_action = self._sampled_actions[info_set_key]
            sampled_action_prob = current_policy.get(sampled_action, 0.0)
            new_opponent_reach = opponent_reach * sampled_action_prob
            self.game.step(sampled_action)
            value = self._update_regrets_dynamic(player, new_opponent_reach)
            self.game.step_back()
        else:
            # Own node: expand all actions full-width
            for action in legal_actions:
                self.game.step(action)
                child_value = self._update_regrets_dynamic(player, opponent_reach)
                self.game.step_back()
                child_values[action] = child_value
                value += current_policy.get(action, 0.0) * child_value

        # Update regrets for own node (unweighted: opponent_reach=1.0 in external sampling)
        if current_player == player:
            for action in legal_actions:
                regret = child_values[action] - value
                self.regret_sum[info_set_key][action] += regret

        # Update strategy sum for opponent node (once per infoset per iteration)
        opponent = 1 - player
        if current_player == opponent and info_set_key not in self._policy_updated_infosets:
            for action in legal_actions:
                action_prob = current_policy.get(action, 0.0)
                self.strategy_sum[info_set_key][action] += action_prob
            self._policy_updated_infosets.add(info_set_key)

        return value

    def _sample_action(self, policy, legal_actions):
        """Sample an action according to policy."""
        actions = list(legal_actions)
        probabilities = np.array([policy.get(a, 0.0) for a in actions], dtype=np.float64)
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total
        else:
            probabilities = np.ones(len(actions), dtype=np.float64) / len(actions)
        action_idx = np.random.choice(len(actions), p=probabilities)
        return actions[action_idx]

    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """No-op: updates occur in _update_regrets_dynamic."""
        pass

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """No-op: updates occur in _update_regrets_dynamic."""
        pass
