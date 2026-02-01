"""Outcome Sampling CFR (dynamic traversal). Sample one terminal history per iteration; regret updates weighted with importance sampling."""

import numpy as np
from typing import Any

from utils.data_models import KeyGenerator
from training.solvers.cfr_solver import CFRSolver
from training.registry import TrainingSolver
from training.config import TrainingConfig


class OutcomeSamplingCFRSolver(CFRSolver):
    """
    Outcome Sampling CFR (dynamic traversal).
    - Per episode: sample one path to terminal (chance and all players sampled).
    - Epsilon-greedy sampling policy for update_player.
    - Importance weighting for regret/strategy updates.
    """

    def __init__(self, game, combination_generator, epsilon=0.6, alternating_updates=False, partial_pruning=False):
        super().__init__(
            game,
            combination_generator,
            alternating_updates=alternating_updates,
            partial_pruning=partial_pruning,
        )
        self.epsilon = epsilon

    @staticmethod
    def evaluate_solver(config: TrainingConfig) -> bool:
        return config.algorithm == 'outcome_sampling'

    @staticmethod
    def create_solver(config: TrainingConfig, game: Any, combo_gen: Any) -> 'TrainingSolver':
        return OutcomeSamplingCFRSolver(game, combo_gen)

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
        """One outcome sampling CFR iteration: one episode per player (one path)."""
        self._current_iteration = self.iteration_count + 1
        for player in range(2):
            self.game.reset(0)
            self._episode_dynamic(update_player=player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

    def _baseline(self, info_set_key, action):
        """Baseline for baseline-corrected outcome sampling (default: 0)."""
        return 0.0

    def _baseline_corrected_child_value(self, sampled_action, action, child_value, sample_prob):
        """Baseline-corrected child value (Eq. 9 Schmid et al. '19)."""
        baseline = self._baseline(None, action)
        if action == sampled_action:
            return baseline + (child_value - baseline) / sample_prob if sample_prob > 1e-10 else baseline
        return baseline

    def _episode_dynamic(self, update_player, my_reach, opp_reach, sample_reach):
        """One episode: one path to terminal, using game API."""
        # Terminal node: return payoff
        if self.game.done:
            return self.game.get_payoff(update_player)

        # Chance node: sample outcome
        if hasattr(self.game, 'is_chance_node') and self.game.is_chance_node():
            outcomes_with_probs = self.game.get_chance_outcomes_with_probs()
            if not outcomes_with_probs:
                return 0.0
            # Sample chance outcome according to probabilities, update reach probabilities
            outcomes = list(outcomes_with_probs.keys())
            weights = np.array([outcomes_with_probs.get(o, 0.0) for o in outcomes], dtype=np.float64)
            s = weights.sum()
            if s <= 0:
                aidx = np.random.randint(len(outcomes))
                p = 1.0 / len(outcomes)
            else:
                weights = weights / s
                aidx = int(np.random.choice(len(outcomes), p=weights))
                p = float(weights[aidx])
            outcome = outcomes[aidx]
            self.game.step(outcome)
            # Recurse with sampled outcome, update opp_reach and sample_reach
            value = self._episode_dynamic(update_player, my_reach, opp_reach * p, sample_reach * p)
            self.game.step_back()
            return value

        # Decision node: get current policy
        current_player = self.game.current_player
        legal_actions = self.game.get_legal_actions()
        info_set_key = KeyGenerator.get_info_set_key(self.game, current_player)
        self.ensure_init(info_set_key, legal_actions)
        current_policy = self.get_current_strategy(info_set_key, legal_actions)
        policy_array = np.array([current_policy.get(a, 0.0) for a in legal_actions], dtype=np.float64)

        # Build epsilon-greedy sampling policy for update_player (epsilon * uniform + (1-epsilon) * current_policy)
        if current_player == update_player:
            uniform_policy = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)
            sample_policy = self.epsilon * uniform_policy + (1.0 - self.epsilon) * policy_array
        else:
            sample_policy = policy_array.copy()

        # Normalize sampling policy
        sample_policy_sum = np.sum(sample_policy)
        if sample_policy_sum > 1e-10:
            sample_policy = sample_policy / sample_policy_sum
        else:
            sample_policy = np.ones(len(legal_actions), dtype=np.float64) / len(legal_actions)

        # Sample action according to sampling policy
        sampled_aidx = int(np.random.choice(len(legal_actions), p=sample_policy))
        sampled_action = legal_actions[sampled_aidx]

        # Update reach probabilities: my_reach for update_player, opp_reach for opponent, sample_reach for sampling
        if current_player == update_player:
            new_my_reach = my_reach * policy_array[sampled_aidx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * policy_array[sampled_aidx]
        new_sample_reach = sample_reach * sample_policy[sampled_aidx]

        # Recurse with sampled action
        self.game.step(sampled_action)
        child_value = self._episode_dynamic(update_player, new_my_reach, new_opp_reach, new_sample_reach)
        self.game.step_back()

        # Compute baseline-corrected child values for all actions
        child_values = {}
        for i, action in enumerate(legal_actions):
            child_values[action] = self._baseline_corrected_child_value(
                sampled_action, action, child_value, sample_policy[i])

        # Estimate state value using current policy and baseline-corrected values
        value_estimate = sum(current_policy.get(a, 0.0) * child_values[a] for a in legal_actions)

        # Update regrets and strategy sum with importance sampling weights (opp_reach/sample_reach, my_reach/sample_reach)
        if current_player == update_player:
            cf_value = value_estimate * opp_reach / sample_reach if sample_reach > 1e-10 else 0.0
            for action in legal_actions:
                cf_action_value = child_values[action] * opp_reach / sample_reach if sample_reach > 1e-10 else 0.0
                regret = cf_action_value - cf_value
                self.regret_sum[info_set_key][action] += regret
            for i, action in enumerate(legal_actions):
                increment = my_reach * policy_array[i] / sample_reach if sample_reach > 1e-10 else 0.0
                self.strategy_sum[info_set_key][action] += increment

        return value_estimate

    def update_regrets(self, info_set_key, legal_actions, action_utilities, current_utility, counterfactual_weight):
        """No-op: updates occur in _episode_dynamic."""
        pass

    def update_strategy_sum(self, info_set_key, legal_actions, current_strategy, player_reach):
        """No-op: updates occur in _episode_dynamic."""
        pass
